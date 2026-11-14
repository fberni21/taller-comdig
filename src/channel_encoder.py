import numpy as np


class ChannelEncoder:
    def __init__(self, generator):
        self.generator = generator
        self.input_bits, self.output_bits = generator.shape

    def _encode_message(self, message):
        return (message @ self.generator) % 2

    def encode(self, bits):
        num_symbols = (len(bits) - 1) // self.input_bits + 1
        pad = self.input_bits * num_symbols - len(bits)
        bits = bits + "0" * pad
        bits = [int(bit) for bit in bits]

        encoded = np.zeros(num_symbols * self.output_bits, dtype=int)

        input_start = 0
        output_start = 0
        for i in range(num_symbols):
            input_end = input_start + self.input_bits
            output_end = output_start + self.output_bits

            encoded[output_start:output_end] =\
                self._encode_message(bits[input_start:input_end])

            input_start = input_end
            output_start = output_end

        encoded = "".join([str(bit) for bit in encoded])

        return encoded, pad
