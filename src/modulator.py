from math import log2, sqrt

import numpy as np


class Modulator:
    def modulate(self, bits):
        # Nota: si len(bits) no es múltiplo de k,
        # se agregan bits de padding (ceros)
        num_symbols = (len(bits) - 1) // self.bits_per_symbol + 1
        pad = self.bits_per_symbol * num_symbols - len(bits)
        bits = bits + "0" * pad

        modulated = np.zeros((num_symbols, self.dimension))

        start = 0
        for i in range(num_symbols):
            end = start + self.bits_per_symbol
            modulated[i] = self.mapping[bits[start:end]]
            start = end

        return modulated, pad

    def compute_mean_energies(self, symbols, code_rate=1.0):
        mean_energy_sym = np.mean(np.sum(symbols**2, axis=1))
        return (mean_energy_sym,
                mean_energy_sym / self.bits_per_symbol / code_rate)


class FskModulator(Modulator):
    def __init__(self, order: int, code_rate=1.0):
        if order not in [2, 4, 8, 16]:
            raise ValueError("'order' must be one of [2, 4, 8, 16]")
        if code_rate <= 0.0 or code_rate > 1.0:
            raise ValueError("'code_rate' must be in range (0, 1]")

        self.order = order
        self.dimension = order
        self.sqrt_code_rate = sqrt(code_rate)
        self.bits_per_symbol = int(log2(order))
        self._build_mapping()

    def _build_mapping(self):
        self.mapping = {}
        for j in range(self.order):
            symbol = np.zeros(self.dimension)
            symbol[j] = self.sqrt_code_rate * sqrt(self.bits_per_symbol)
            self.mapping[f'{j:0{self.bits_per_symbol}b}'] = symbol


class QamModulator(Modulator):
    GRAY = {4: ['00', '01', '10', '11'],
            16: [f'{j:04b}' for j in [0,  1,  3,  2,
                                      4,  5,  7,  6,
                                      12, 13, 15, 14,
                                      8,  9,  11, 10]]}

    def __init__(self, order: int, code_rate=1.0):
        if order not in [4, 16]:
            raise ValueError("'order' must be one of [4, 16]")
        if code_rate <= 0.0 or code_rate > 1.0:
            raise ValueError("'code_rate' must be in range (0, 1]")

        self.order = order
        self.dimension = 2
        self.sqrt_code_rate = sqrt(code_rate)
        self.bits_per_symbol = int(log2(order))
        self._build_mapping()

    def _build_mapping(self):
        self.mapping = {}
        side = int(sqrt(self.order))

        for j in range(self.order):
            # Para energía media de bit de información unitaria:
            d = sqrt(6 * self.bits_per_symbol / (self.order - 1))
            symbol = self.sqrt_code_rate * (
                    (np.array([j // side, j % side]) - 0.5*(side - 1)) * d)
            self.mapping[QamModulator.GRAY[self.order][j]] = symbol
