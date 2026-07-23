import json
from pathlib import Path
import random

def generate_synthetic_prompts(
        num_prompts: int,
        prompt_len_range: tuple[int, int],
        tokenizer, 
        seed: int = 42,
) -> list[str]:
    """生成 synthetic prompts (M1-M2 用)"""
    random.seed(seed)

    base_prompts = [
        "Tell me about artificial intelligence and its impact on society.",
        "Explain quantum computing in simple terms.",
        "What are the benefits of regular exercise?",
        "How does photosynthesis work?",
        "Describe the history of the internet.",
    ]

    # 通过重复和截断生成不同长度
    prompts = []

    for i in range(num_prompts):
        base = base_prompts[i % len(base_prompts)] 
        target_len = random.randint(*prompt_len_range)  

        # 重复 base 直到达到目标长度
        current = base 
        while len(tokenizer.encode(current)) < target_len: 
            current = current + " " + base 
        
        # 截断到目标长度
        tokens = tokenizer.encode(current)[:target_len] 
        prompt = tokenizer.decode(tokens)
        prompts.append(prompt)
    
    return prompts 

def load_sharegpt(
    num_prompts: int,
    tokenizer,
    max_model_len: int = 4096,
    seed: int = 42,
    max_output_len: int | None = None,
) -> list[tuple[str, int]]:
    """Load real ShareGPT conversations as (prompt_text, output_len) request specs.

    output_len is the tokenized length of the *actual* assistant reply, which becomes that
    request's max_tokens. This is the single most important realism fix for the M4
    benchmark: until now every request ran to one globally hardcoded output_len, but real
    traffic has a *distribution* of response lengths. Throughput and goodput only mean
    something when each request runs to its own length.

    Filtering follows the shape of vLLM's benchmark_serving.py::sample_sharegpt_requests.

    Args:
        num_prompts: how many specs to return.
        tokenizer: used to measure prompt and reply lengths.
        max_model_len: drop conversations whose prompt + reply cannot fit the context.
        seed: fixed seed makes the sample deterministic across runs.
        max_output_len: optional cap so a few very long replies do not dominate a run.

    Raises:
        RuntimeError: if fewer than num_prompts conversations survive filtering.
    """
    from datasets import load_dataset

    ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")

    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)

    specs: list[tuple[str, int]] = []
    for idx in indices:
        if len(specs) >= num_prompts:
            break

        conversations = ds[idx].get("conversations") or []
        if len(conversations) < 2:
            continue
        # Use the opening human turn as the prompt and the reply that follows as the
        # reference completion; anything else is not a clean request/response pair.
        if conversations[0].get("from") != "human" or conversations[1].get("from") != "gpt":
            continue

        prompt, answer = conversations[0]["value"], conversations[1]["value"]
        prompt_len = len(tokenizer.encode(prompt))
        output_len = len(tokenizer.encode(answer))
        if max_output_len is not None:
            output_len = min(output_len, max_output_len)

        if prompt_len < 4 or output_len < 4:
            continue  # degenerate turns measure nothing
        if prompt_len + output_len > max_model_len:
            continue  # would not fit in the context window

        specs.append((prompt, output_len))

    if len(specs) < num_prompts:
        raise RuntimeError(
            f"only {len(specs)} conversations survived filtering, need {num_prompts}"
        )
    return specs


# --------------------------------------------------------------------------------------
# Spec persistence -- the key to a fair nano-vs-vLLM comparison.
#
# Neither engine should sample its own prompts: vLLM's benchmark_serving.py re-samples and
# re-filters ShareGPT internally, so it would never pick the same 64 conversations that
# load_sharegpt(seed=42) picks. Instead we materialise the specs once, to disk, and have
# *both* backends replay that exact file. Then the only thing that differs between the two
# runs is the engine.
# --------------------------------------------------------------------------------------
def dump_specs(specs: list[tuple[str, int]], path: str) -> None:
    """Write (prompt, output_len) specs to JSON so both backends can replay them verbatim."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([{"prompt": p, "output_len": n} for p, n in specs], f)


def load_specs(path: str) -> list[tuple[str, int]]:
    """Load specs previously written by dump_specs, preserving order."""
    with open(path) as f:
        return [(item["prompt"], item["output_len"]) for item in json.load(f)]