import torch
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import seaborn as sns


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=16,
        help="",
    )
    parser.add_argument(
        "--max-seqlen",
        type=int,
        default=64,
        help="",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="",
    )
    parser.add_argument(
        "--bench-backward",
        action="store_true",
    )
    parser.add_argument(
        "--bench-generate",
        action="store_true",
    )
    parser.add_argument(
        "--use-padding",
        action="store_true",
    )
    return parser

model_id = "meta-llama/Llama-2-7b-hf"

@torch.no_grad()
def warmup_and_benchmark(
    model,
    batch_size,
    max_seq_len,
    padding_ratio: float,
    num_batches,
    bench_generate,
    bench_backward,
    max_new_tokens,
):
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, max_seq_len)).to(0)
    inputs = {"input_ids": input_ids}


    if padding_ratio != 0 and batch_size > 1:
        attention_mask = torch.zeros_like(input_ids)
        last_valid_token = int((1 - padding_ratio) * max_seq_len)
        print("last_valid_token", last_valid_token)
        attention_mask[:, :last_valid_token] = 1
        attention_mask[0, :] = 1
        inputs["attention_mask"] = attention_mask
    else:
        inputs["attention_mask"] = torch.ones_like(input_ids)

    model.generation_config.eos_token_id = None

    # warmup
    if bench_generate:
        _ = model.generate(**inputs, min_new_tokens=max_new_tokens, max_new_tokens=max_new_tokens, eos_token_id=None, use_cache=True)
    else:
        _ = model(**inputs, use_cache=False)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    with torch.no_grad():
        start_event.record()
        for _ in range(num_batches):
            if bench_generate:
                _ = model.generate(**inputs, min_new_tokens=max_new_tokens, max_new_tokens=max_new_tokens, eos_token_id=None, use_cache=True)
            else:
                _ = model(**inputs, use_cache=False)
        end_event.record()
        torch.cuda.synchronize()

    forward_timing = (start_event.elapsed_time(end_event) * 1.0e-3) / num_batches
    backward_timing = 0

    if bench_backward:
        for _ in range(num_batches):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            logits = model(input_ids).logits
            loss = logits.mean()

            start_event.record()
            loss.backward()

            end_event.record()
            torch.cuda.synchronize()

            backward_timing += (start_event.elapsed_time(end_event) * 1.0e-3)

    return forward_timing, backward_timing / num_batches

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    num_batches = args.num_batches
    max_seq_len = args.max_seqlen
    max_batch_size = args.max_batch_size
    max_new_tokens = args.max_new_tokens

    bench_generate = args.bench_generate
    bench_backward = args.bench_backward

    if args.use_padding:
        padding_ratio = 0.3
    else:
        padding_ratio = 0

    if not args.bench_generate:
        max_new_tokens = 1

    # TODO: change this
    BATCH_SIZE = [1, max_batch_size // 4, max_batch_size // 2, max_batch_size]

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.float16
    )
    print("model", model)

    print("LOAD FLASH")
    model_fa = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.float16,
        use_flash_attn_2=True
    )
    print("model_fa", model_fa)

    native_total_time_dict = {}
    fa2_total_time_dict = {}
    forward_speedups = {}
    backward_speedups = {}

    for batch_size in tqdm(BATCH_SIZE):
        print("Timing vanilla")
        native_timing, native_backward_timing = warmup_and_benchmark(
            model,
            batch_size,
            max_seq_len,
            padding_ratio,
            num_batches,
            bench_generate,
            bench_backward,
            max_new_tokens
        )
        native_total_time_dict[f"{batch_size}"] = native_timing

        print("Timing FA")
        fa2_timing, fa2_backward_timing = warmup_and_benchmark(
            model_fa,
            batch_size,
            max_seq_len,
            padding_ratio,
            num_batches,
            bench_generate,
            bench_backward,
            max_new_tokens
        )
        fa2_total_time_dict[f"{batch_size}"] = fa2_timing

        forward_speedups[f"{batch_size}"] = native_timing / fa2_timing
        if bench_backward:
            backward_speedups[f"{batch_size}"] = native_backward_timing / fa2_backward_timing
        else:
            backward_speedups[f"{batch_size}"] = 0

    dir_name = f"flash-attn-2-benchmarks/{model_id}/seq_len_{max_seq_len}_padding_{padding_ratio}_generate_{bench_generate}_max_batch_size_{max_batch_size}/"
    os.makedirs(dir_name, exist_ok=True)

    sns.set(style="darkgrid")
    # plot both lines
    sns.lineplot(data=native_total_time_dict, color="blue", label="llama2-native")
    sns.lineplot(data=fa2_total_time_dict, color="orange", label="llama2-FA2")

    plt.ylabel("Average inference time (s)")
    plt.xlabel("Batch size")
    plt.title("Comparing average inference time between native model vs Flash Attention-2 model - ran on NVIDIA A100", fontsize = 8)
    plt.suptitle(f"prompt max length: {max_seq_len} | new tokens : {max_new_tokens} | generate: {bench_generate} | padding ratio: {padding_ratio} - ", fontsize = 8)

    plt.legend()

    # save plot
    plt.savefig(os.path.join(dir_name, "timing_plot.jpg"), dpi=300)

    plt.figure()
    sns.set(style="darkgrid")
    # plot both lines
    sns.lineplot(data=forward_speedups, color="orange", label="forward-speedup")
    if bench_backward:
        sns.lineplot(data=backward_speedups, color="blue", label="backward-speedup")

    plt.ylabel("Speedup (x)")
    plt.xlabel("Batch size")
    plt.title("Comparing forward/backward speedup between native model vs Flash Attention-2 model - ran on NVIDIA A100", fontsize = 8)
    plt.suptitle(f"prompt max length: {max_seq_len} | new tokens : {max_new_tokens} | generate: {bench_generate} | padding ratio: {padding_ratio} - ", fontsize = 8)

    plt.legend()

    # save plot
    plt.savefig(os.path.join(dir_name, "speedup_plot.jpg"), dpi=300)
