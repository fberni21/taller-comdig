import numpy as np


class Channel:
    def __init__(self, N0=1):
        self.N0 = N0

    def transmit(self, symbols):
        symbols_noisy = np.random.normal(0, np.sqrt(self.N0/2), size=symbols.shape)
        return symbols + symbols_noisy
