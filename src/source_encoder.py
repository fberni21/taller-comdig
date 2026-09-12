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
    # Construye un árbol cuya forma final tiene en las hojas a los
    # diferentes caracteres. Los caracteres más probables quedan a
    # alturas más chicas, y los menos probables a alturas grandes.
    # Es una forma estructurada de realizar el algoritmo de Huffman.

    # Inicialmente, tenemos los nodos sueltos y son todos hojas
    # (tienen un caracter `c` asociado, y sus hijos son None).
    tree = [(probs[c], c, None, None) for c in range(256)]

    # El min-heap se usa para poder extraer los nodos de menor
    # probabilidad en forma eficiente
    heapq.heapify(tree)

    while len(tree) != 1:
        # Tomamos los dos nodos de menor probabilidad, y los hacemos
        # hijos de un nuevo nodo cuya probabilidad es la suma de las
        # probabilidades de sus hijos. Este nuevo nodo no tiene un
        # caracter asociado (denotado por `-1`).
        t1 = heapq.heappop(tree)
        t2 = heapq.heappop(tree)
        new = (t1[0] + t2[0], -1, t1, t2)
        heapq.heappush(tree, new)

    # La raíz del árbol es el único elemento
    return tree[0]


def huff_generate_code(node, b, code_map):
    # Recorre recursivamente el árbol de Huffman, para asignar los
    # códigos correspondientes.
    #
    # Las variables `b` y `code_map` inicialmente están vacías.
    # - `b` acumula el código de una rama.
    # - `code_map` es un diccionario donde se agregan todos los códigos.

    if node is None:
        return code_map

    # Recorremos el sub árbol izquierdo, extendiendo con un `0`.
    code_map = huff_generate_code(node[2], b + '0', code_map)

    # Si este nodo es hoja, le asignamos el código acumulado en `b`.
    if node[1] != -1:
        code_map[node[1]] = b[:]

    # Recorremos el sub árbol derecho, extendiendo con un `1`.
    return huff_generate_code(node[3], b + '1', code_map)


def huff_encode(probs):
    tree = huff_build_tree(probs)
    code_map = huff_generate_code(tree, '', {})
    return code_map


class SourceEncoder:
    pass
