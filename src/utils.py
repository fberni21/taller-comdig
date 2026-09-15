import numpy as np


def compute_char_probs(text):
    probs = np.zeros(256)
    text_len = len(text)
    unique, counts = np.unique(text, return_counts=True)
    for char, count in zip(unique, counts):
        probs[char] = count / text_len
    return probs


def compute_entropy(probs):
    return -np.sum(probs * np.log2(probs + 1e-10))


def is_prefix_code(code_map):
    for i, this in enumerate(code_map):
        for that in code_map[i + 1:]:
            if this.startswith(that) or that.startswith(this):
                return False
    return True
