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

    # E.2: Generación de atenuación uniforme
    def generate_attenuation(self, num_symbols):
        return self.rng.uniform(
            low=0.5,
            high=0.9,
            size=num_symbols
        )

    # E.3: Aplicación de los efectos del canal
    def transmit(self, symbols):
        symbols = np.asarray(symbols)

        noise = self.generate_noise(symbols.shape)
        attenuation = self.generate_attenuation(
            symbols.shape[0]
        )

        # Una atenuación por símbolo, compartida
        # por todas sus componentes.
        received = attenuation[:, None] * symbols + noise

        return received
