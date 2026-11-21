# hw1-build: caspian-7

import json
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.merges = []
        self.vocab_size = 256
        self.trained = False

    def train(self, text: str, vocab_size: int) -> None:
        if not 256 <= vocab_size <= 65536:
            raise ValueError("vocab_size must be between 256 and 65536")

        tokens = list(text.encode("utf-8"))
        self.merges = []

        for _ in range(vocab_size - 256):
            if len(tokens) < 2:
                break

            pair_counts = Counter(zip(tokens, tokens[1:]))
            max_count = max(pair_counts.values())

            best_pair = None
            for pair in zip(tokens, tokens[1:]):
                if pair_counts[pair] == max_count:
                    best_pair = pair
                    break

            a, b = best_pair
            new_id = 256 + len(self.merges)
            self.merges.append((a, b))

            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == a
                    and tokens[i + 1] == b
                ):
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        self.vocab_size = vocab_size
        self.trained = True

    def encode(self, text: str) -> list[int]:
        if not self.trained:
            raise ValueError("Tokenizer has not been trained")

        tokens = list(text.encode("utf-8"))

        for k, (a, b) in enumerate(self.merges):
            new_id = 256 + k
            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == a
                    and tokens[i + 1] == b
                ):
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def decode(self, ids: list[int]) -> str:
        vocab = {i: bytes([i]) for i in range(256)}

        for k, (a, b) in enumerate(self.merges):
            vocab[256 + k] = vocab[a] + vocab[b]

        result = b"".join(vocab[i] for i in ids)
        return result.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
            "trained": self.trained
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.vocab_size = data["vocab_size"]
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer.trained = data["trained"]

        return tokenizer
