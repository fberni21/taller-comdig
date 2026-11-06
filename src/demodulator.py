import numpy as np


class Demodulator:
    def __init__(self, mapping):
        # `mapping` es el diccionario {etiqueta: punto} del modulador.
        self.mapping = mapping
        # Separarlo en dos estructuras alineadas por índice:
        self.etiquetas = list(mapping.keys()) 
        self.puntos_constelacion = np.array(list(mapping.values()))

    def demodulate(self, received, pad):

        N = received.shape[0]
        mejor_distancia = np.full(N, np.inf)        # arranca en infinito: cualquier distancia va a ser menor
        mejor_indice = np.zeros(N, dtype=int)       # el índice del punto más cercano encontrado


        for i in range(self.puntos_constelacion.shape[0]):
            # estoy iterando por cada uno de los puntos de la constelación.
            D = np.linalg.norm(received - self.puntos_constelacion[i], axis=1)

            mask = D < mejor_distancia
            
            mejor_distancia[mask] = D[mask]
            mejor_indice[mask] = i

        bits = ''.join([self.etiquetas[i] for i in mejor_indice])
        bits = bits[:len(bits)-pad]

        puntos_decididos = self.puntos_constelacion[mejor_indice]   # (N, dim)

        return bits, puntos_decididos
