import numpy as np
import heapq


def compute_char_probs(text):
    char_counts = np.zeros(256)
    unique, counts = np.unique(list(text), return_counts=True)
    for char, count in zip(unique, counts):
        char_counts[ord(char)] = count
    return char_counts / len(text)


def compute_entropy(probs):
    return -np.sum(probs * np.log2(probs + 1e-10))


def huff_build_tree(probs):
    tree = [(probs[c], c, None, None) for c in range(256)]
    heapq.heapify(tree)
    while len(tree) != 1:
        t1 = heapq.heappop(tree)
        t2 = heapq.heappop(tree)
        new = (t1[0] + t2[0], -1, t1, t2)
        heapq.heappush(tree, new)
    return tree


def huff_generate_code(node, b, code_map):
    if node is None:
        return code_map
    code_map = huff_generate_code(node[2], b + '0', code_map)
    if node[1] != -1:
        code_map[node[1]] = b[:]
    return huff_generate_code(node[3], b + '1', code_map)


def huff_encode(probs):
    tree = huff_build_tree(probs)
    code_map = huff_generate_code(tree[0], '', {})
    return code_map


class SourceEncoder:
    pass
