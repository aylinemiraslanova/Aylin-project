# hw1-build: caspian-7

import os
import json
import math
import time
import csv
import argparse
import torch

from model import GPT, GPTConfig


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(project_dir, device):
    train_tokens = torch.load(
        os.path.join(project_dir, "cache", "train_tokens.pt"),
        map_location="cpu",
        weights_only=True
    )

    val_tokens = torch.load(
        os.path.join(project_dir, "cache", "val_tokens.pt"),
        map_location="cpu",
        weights_only=True
    )

    return train_tokens, val_tokens


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size] for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1] for i in ix
    ])

    return x.to(device), y.to(device)


def get_lr(step, max_lr, warmup_steps, max_steps):
    min_lr = max_lr * 0.1

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step >= max_steps:
        return min_lr

    decay_ratio = (
        (step - warmup_steps) /
        (max_steps - warmup_steps)
    )

    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, train_tokens, val_tokens,
                  batch_size, block_size, device,
                  eval_batches=50):

    model.eval()
    results = {}

    for split, data in [
        ("train", train_tokens),
        ("val", val_tokens)
    ]:
        losses = []

        for _ in range(eval_batches):
            x, y = get_batch(
                data, batch_size, block_size, device
            )

            _, loss = model(x, y)
            losses.append(loss.item())

        results[split] = sum(losses) / len(losses)

    model.train()
    return results


def save_log(log_path, rows):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "train_loss",
                "val_loss",
                "lr",
                "tokens_seen"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)


def train(config, project_dir):

    seed = config["seed"]
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tokens, val_tokens = load_data(project_dir, device)

    model_config = GPTConfig(
        vocab_size=1024,
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
        use_positional_embeddings=config.get("use_positional_embeddings", True),
        norm_style=config.get("norm_style", "pre")
    )

    model = GPT(model_config).to(device)

    decay_params = []
    no_decay_params = []

    for param in model.parameters():
        if not param.requires_grad:
            continue

        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0}
        ],
        lr=config["max_lr"],
        betas=(0.9, 0.95)
    )

    max_steps = config["max_steps"]
    eval_interval = config["eval_interval"]
    batch_size = config["batch_size"]
    block_size = config["block_size"]
    max_lr = config["max_lr"]

    # Assignment requires exactly 137 warmup steps
    warmup_steps = 137

    best_val_loss = float("inf")
    log_rows = []

    checkpoint_dir = os.path.join(project_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{config['run_id']}_best.pt"
    )

    start_time = time.time()

    for step in range(max_steps):

        lr = get_lr(
            step, max_lr, warmup_steps, max_steps
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if step % eval_interval == 0:
            losses = estimate_loss(
                model,
                train_tokens,
                val_tokens,
                batch_size,
                block_size,
                device
            )

            train_loss = losses["train"]
            val_loss = losses["val"]

            tokens_seen = step * batch_size * block_size

            print(
                f"Step {step}: "
                f"train={train_loss:.4f}, "
                f"val={val_loss:.4f}, "
                f"lr={lr:.6f}"
            )

            log_rows.append({
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "tokens_seen": tokens_seen
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss
                }, checkpoint_path)

        model.train()

        x, y = get_batch(
            train_tokens, batch_size, block_size, device
        )

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )

        optimizer.step()

    elapsed = time.time() - start_time

    # Final evaluation
    final_losses = estimate_loss(
        model, train_tokens, val_tokens,
        batch_size, block_size, device
    )

    final_train_loss = final_losses["train"]
    final_val_loss = final_losses["val"]

    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": max_steps,
            "best_val_loss": best_val_loss
        }, checkpoint_path)

    # Save log
    log_path = os.path.join(
        project_dir, "logs", f"{config['run_id']}.csv"
    )
    save_log(log_path, log_rows)

    # Run statistics
    params = model.num_params()
    tokens_seen = max_steps * batch_size * block_size
    tokens_per_sec = tokens_seen / elapsed

    # Save runs.csv
    runs_path = os.path.join(project_dir, "runs.csv")

    run_row = {
        "run_id": config["run_id"],
        "n_layer": config["n_layer"],
        "n_head": config["n_head"],
        "n_embd": config["n_embd"],
        "block_size": block_size,
        "max_lr": max_lr,
        "max_steps": max_steps,
        "ablation": config["ablation"],
        "params": params,
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "tokens_per_sec": tokens_per_sec,
        "wall_time_min": elapsed / 60
    }

    fieldnames = list(run_row.keys())
    file_exists = os.path.exists(runs_path)

    with open(runs_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(run_row)

    print("\\nTraining completed!")
    print("Best validation loss:", best_val_loss)
    print("Final train loss:", final_train_loss)
    print("Parameters:", params)
    print("Tokens/sec:", round(tokens_per_sec, 2))
    print("Wall time:", round(elapsed / 60, 2), "minutes")

    return model, best_val_loss, elapsed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(
        project_dir, args.config
    )

    config = load_config(config_path)

    print("Starting run:", config["run_id"])

    model, best_val_loss, elapsed = train(
        config, project_dir
    )


if __name__ == "__main__":
    main()
