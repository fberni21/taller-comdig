
import numpy as np
from itertools import combinations, product


class ChannelDecoder:
    def __init__(self, generator):
        self.generator = np.asarray(generator, dtype=int)
        self.input_bits, self.output_bits = self.generator.shape

        self.parity_check = self._calculate_H(
            self.input_bits, self.output_bits, self.generator
        )
        self.syndrome_table = self._calculate_syndromes(self.parity_check)

    # Inciso 1: calcular la matriz de chequeo de paridad H
    def _calculate_H(self, k, n, G):
        G = np.asarray(G, dtype=int)

        if G.shape != (k, n):
            raise ValueError("Las dimensiones de G no coinciden con k y n.")

        r = n - k

        # La matriz generadora tiene la forma G = [P | I_k]
        P = G[:, :r]

        if not np.array_equal(G[:, r:], np.eye(k, dtype=int)):
            raise ValueError("La matriz G debe tener la forma [P | I_k].")

        Id = np.eye(r, dtype=int)

        # H = [I_(n-k) | P.T]
        H = np.concatenate((Id, P.T), axis=1) % 2

        if not np.all((G @ H.T) % 2 == 0):
            raise ValueError("La matriz H calculada no es valida.")

        return H

    # Inciso 2: calcular la tabla de sindromes S
    def _calculate_syndromes(self, H):
        H = np.asarray(H, dtype=int)

        r, n = H.shape
        table = {}

        # Se prueban errores de menor a mayor peso
        for weight in range(n + 1):
            for positions in combinations(range(n), weight):
                error = np.zeros(n, dtype=int)
                error[list(positions)] = 1

                syndrome = tuple((H @ error) % 2)

                if syndrome not in table:
                    table[syndrome] = error

                # Se encontraron todos los sindromes posibles
                if len(table) == 2 ** r:
                    return table

        return table

    # Inciso 3: detectar y corregir una palabra recibida
    def _decode_word(self, H, S, word):
        # Calcular el sindrome de la palabra recibida
        syndrome = tuple((H @ word) % 2)

        error = S[syndrome]

        # Corregir la palabra recibida
        corrected = (word + error) % 2

        return corrected

    # Inciso 4: decodificar una secuencia de bits
    def decode(self, H, S, bits, pad=0):
        H = np.asarray(H, dtype=int)

        r, n = H.shape
        k = n - r

        if len(bits) % n != 0:
            raise ValueError(
                "La longitud recibida debe ser multiplo de n."
            )

        num_symbols = len(bits) // n
        decoded = np.zeros(num_symbols * k, dtype=int)

        input_start = 0
        output_start = 0

        bits = [int(bit) for bit in bits]

        for i in range(num_symbols):
            input_end = input_start + n
            output_end = output_start + k

            # Tomar una palabra de n bits
            word = bits[input_start:input_end]

            # Detectar y corregir utilizando el inciso 3
            corrected = self._decode_word(H, S, word)

            # Recuperar los k bits de informacion
            # G = [P | I_k]
            decoded[output_start:output_end] = corrected[-k:]

            input_start = input_end
            output_start = output_end

        decoded = "".join(str(bit) for bit in decoded)

        # Eliminar los bits de relleno del codificador
        if pad:
            if not 0 <= pad <= len(decoded):
                raise ValueError("Cantidad de bits de relleno invalida.")
            decoded = decoded[:-pad]

        return decoded

    # Inciso 5: calcular dmin, e y t
    def code_parameters(self, G):
        G = np.asarray(G, dtype=int)
        k, n = G.shape

        dmin = n + 1

        # Generar todos los mensajes posibles de k bits
        for message in product([0, 1], repeat=k):
            message = np.array(message, dtype=int)

            # No considerar el mensaje nulo
            if not np.any(message):
                continue

            # Codificar el mensaje
            codeword = (message @ G) % 2

            # Calcular su peso de Hamming
            weight = np.sum(codeword)

            # Buscar el menor peso no nulo
            if 0 < weight < dmin:
                dmin = weight

        if dmin == n + 1:
            raise ValueError("No se encontraron palabras no nulas.")

        # Errores detectables
        e = dmin - 1

        # Errores corregibles garantizados
        t = (dmin - 1) // 2

        return dmin, e, t
