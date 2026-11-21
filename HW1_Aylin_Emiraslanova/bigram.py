# hw1-build: caspian-7

import torch


def evaluate_bigram(train_tokens, val_tokens, vocab_size=1024):
    pair_ids = (
        train_tokens[:-1] * vocab_size
        + train_tokens[1:]
    )

    counts = torch.bincount(
        pair_ids,
        minlength=vocab_size * vocab_size
    ).reshape(vocab_size, vocab_size)

    probs = (counts + 1).float()

    probs = probs / (
        counts.sum(dim=1, keepdim=True) + vocab_size
    )

    val_prev = val_tokens[:-1]
    val_next = val_tokens[1:]

    selected_probs = probs[val_prev, val_next]

    loss = -torch.log(selected_probs).mean().item()

    return loss


if __name__ == "__main__":
    print("Bigram module loaded successfully.")
