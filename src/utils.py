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
