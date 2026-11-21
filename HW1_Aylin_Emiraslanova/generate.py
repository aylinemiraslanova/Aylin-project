# hw1-build: caspian-7


import torch
import torch.nn.functional as F


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None, generator=None):

    if temperature < 0:
        raise ValueError("Temperature cannot be negative")

    # Greedy decoding
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    # Top-K filtering
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        threshold = values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # Top-P filtering
    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )

        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs - sorted_probs >= top_p
        sorted_logits = sorted_logits.masked_fill(
            sorted_indices_to_remove, float("-inf")
        )

        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(1, sorted_indices, sorted_logits)
        logits = filtered_logits

    probs = F.softmax(logits, dim=-1)

    return torch.multinomial(
        probs, num_samples=1, generator=generator
    )



def generate(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
    seed: int | None = None
) -> list[int]:

    if not prompt_ids:
        raise ValueError("Prompt cannot be empty")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")

    device = next(model.parameters()).device
    block_size = model.config.block_size

    if len(prompt_ids) > block_size:
        raise ValueError("Prompt exceeds block_size")

    generator = None

    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    model.eval()

    idx = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    past_kv = None

    with torch.no_grad():

        for _ in range(max_new_tokens):

            if use_cache:

                if past_kv is None:
                    input_ids = idx
                else:
                    input_ids = idx[:, -1:]

                logits, _, past_kv = model(
                    input_ids,
                    past_kv=past_kv,
                    use_cache=True
                )

            else:
                input_ids = idx[:, -block_size:]
                logits, _ = model(input_ids)

            next_token_logits = logits[:, -1, :]

            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator
            )

            idx = torch.cat((idx, next_token), dim=1)

            if use_cache and idx.size(1) >= block_size:
                if _ != max_new_tokens - 1:
                    raise ValueError(
                        "Generation exceeds block_size with KV cache"
                    )

    return idx[0].tolist()

