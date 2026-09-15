import heapq

import numpy as np


class HuffmanEncoder:
    INNER = 256

    def __init__(self, probs):
        self.probs = probs
        self.code_map = [None] * 256
        self.lengths = None
        self.length_avg = None
        self.length_var = None

        self._build_map()

    def _compute_stats(self):
        self.lengths = np.array([len(code) for code in self.code_map])
        self.length_avg = np.sum(self.probs * self.lengths)
        self.length_var = np.sum(
                self.probs * (self.lengths - self.length_avg)**2)

    def _build_map(self):
        self._build_tree()
        self._generate_codes(self.tree, '')
        self._compute_stats()

    def _build_tree(self):
        # Construye un árbol cuya forma final tiene en las hojas a los
        # diferentes caracteres. Los caracteres más probables quedan a
        # alturas más chicas, y los menos probables a alturas grandes.
        # Es una forma estructurada de realizar el algoritmo de Huffman.

        # Inicialmente, tenemos los nodos sueltos y son todos hojas
        # (tienen un caracter `char` asociado, y sus hijos son None).
        nodes = [(prob, char, None, None)
                 for char, prob in enumerate(self.probs)]

        # El min-heap se usa para poder extraer los nodos de menor
        # probabilidad en forma eficiente
        heapq.heapify(nodes)

        while len(nodes) != 1:
            # Tomamos los dos nodos de menor probabilidad, y los hacemos
            # hijos de un nuevo nodo cuya probabilidad es la suma de las
            # probabilidades de sus hijos. Este nuevo nodo no tiene un
            # caracter asociado.
            t1 = heapq.heappop(nodes)
            t2 = heapq.heappop(nodes)
            new = (t1[0] + t2[0], self.INNER, t1, t2)
            heapq.heappush(nodes, new)

        # La raíz del árbol es el único elemento
        self.tree = nodes[0]

    def _generate_codes(self, node, b):
        # Recorre recursivamente el árbol de Huffman, para asignar los
        # códigos correspondientes.
        #
        # Las variables `b` y `code_map` inicialmente están vacías.
        # - `b` acumula el código de una rama.
        # - `code_map` es un diccionario donde se agregan todos los códigos.

        if node is None:
            return

        # Recorremos el sub árbol izquierdo, extendiendo con un `0`.
        self._generate_codes(node[2], b + '0')

        # Si este nodo es hoja, le asignamos el código acumulado en `b`.
        if node[1] != self.INNER:
            self.code_map[node[1]] = b[:]

        # Recorremos el sub árbol derecho, extendiendo con un `1`.
        self._generate_codes(node[3], b + '1')


class SourceEncoder:
    def __init__(self, code_map):
        self.code_map = code_map

    def encode(self, text):
        return ''.join([self.code_map[c] for c in text])


class SourceDecoder:
    def __init__(self, code_map):
        self.decode_map = {v: chr(k) for k, v in enumerate(code_map)}

    def decode(self, encoded):
        length = len(encoded)
        decoded = []
        start = 0
        while start < length:
            end = start + 1
            while end < length and encoded[start:end] not in self.decode_map:
                end += 1
            decoded.append(self.decode_map[encoded[start:end]])
            start = end
        return ''.join(decoded)
