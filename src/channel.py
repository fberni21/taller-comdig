import numpy as np


class Channel:
    def __init__(self, N0=1, seed=None):
        self.N0 = N0
        self.rng = np.random.default_rng(seed)

    # E.1: Generación de ruido AWGN
    def generate_noise(self, shape):
        sigma = np.sqrt(self.N0 / 2)

        return self.rng.normal(
            loc=0,
            scale=sigma,
            size=shape
        )

    # E.3: Aplicación de los efectos del canal
    def transmit(self, symbols):
        symbols = np.asarray(symbols)
        noise = self.generate_noise(symbols.shape)
        received = symbols + noise
        return received
