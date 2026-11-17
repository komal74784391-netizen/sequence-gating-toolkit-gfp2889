"""Sequence gating toolkit: small, generic utilities."""


def build_attention_mask(length, window=None):
    mask = [[1] * length for _ in range(length)]
    if window:
        for i in range(length):
            for j in range(length):
                if abs(i - j) > window:
                    mask[i][j] = 0
    return mask


def pad_batch(sequences, pad_id=0):
    width = max(len(s) for s in sequences)
    return [s + [pad_id] * (width - len(s)) for s in sequences]


def sigmoid(x):
    import math
    return 1.0 / (1.0 + math.exp(-x))
