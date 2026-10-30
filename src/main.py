import argparse

import matplotlib.pyplot as plt
import numpy as np

import source
from modulator import FskModulator, QamModulator

from utils import compute_char_probs, compute_entropy, is_prefix_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='input file name')
    parser.add_argument('mod_scheme', choices=['QAM', 'FSK'],
                        help='modulation scheme')
    parser.add_argument('mod_order', choices=[2, 4, 8, 16], type=int,
                        help='modulation order')
    parser.add_argument('--sample', default=None,
                        help='sample line to encode and decode')
    parser.add_argument('--output', default='decoded.txt',
                        help='decoded file name (default: decoded.txt)')
    parser.add_argument('--plot-constellation',
                        action=argparse.BooleanOptionalAction,
                        help='plot constellation (only for 2D constellations)')
    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        text = file.read()

    chars = [ord(c) for c in text]
    chars_count = len(chars)

    probs = compute_char_probs(chars)
    entropy = compute_entropy(probs)

    huff = source.HuffmanEncoder(probs)

    print(f'File entropy              : {entropy:.4f} bits/symbol')
    print(f'Minimum length            : {entropy:.4f} bits/symbol')
    print(f'Code mean length          : {huff.length_avg:.4f} bits/symbol')
    print(f'Code variance             : {huff.length_var:.4f} (bits/symbol)^2')
    print(f'Code efficiency           : {100*(entropy/huff.length_avg):.2f} %')
    print(f'Fixed code (ASCII) length : {8} bits/symbol')
    print(f'Is prefix code?           : {is_prefix_code(huff.code_map)}')
    print()

    print('=' * 72)
    print('Char\tCounts\tProbability\tCode')
    for k in sorted(range(256), key=lambda k: probs[k], reverse=True):
        p = probs[k]
        if p == 0:
            continue
        count = round(p * chars_count)
        print(f'{chr(k)!r}\t{count}\t{p:.8f}\t{huff.code_map[k]}')
    print()

    source_enc = source.SourceEncoder(huff.code_map)
    source_dec = source.SourceDecoder(huff.code_map)

    print('=' * 72)

    sample = args.sample
    if sample is not None:
        sample_chars = [ord(c) for c in sample]
        sample_encoded = source_enc.encode(sample_chars)
        sample_decoded = source_dec.decode(sample_encoded)
        print(f'Sample         : {sample!r}')
        print(f'Encoded sample : {sample_encoded}')
        print(f'Decoded sample : {sample_decoded!r}')
        print(f'Sample length  : {len(sample_chars)} symbols')
        print(f'Encoded length : {len(sample_encoded)} bits')
        print(f'Output equal?  : {sample == sample_decoded}')
        print()

    encoded = source_enc.encode(chars)
    decoded = source_dec.decode(encoded)
    encoded_length = len(encoded)
    ascii_length = chars_count * 8
    compression_savings = 1 - encoded_length / ascii_length

    with open(args.output, 'w') as f:
        f.write(decoded)

    print('=' * 72)
    print(f'Character count       : {chars_count} symbols')
    print(f'Bits needed (Huffman) : {len(encoded)} bits')
    print(f'Bits needed (ASCII)   : {chars_count * 8} bits')
    print(f'Compression savings   : {100 * compression_savings:.2f} %')
    print(f'Output equal?         : {text == decoded}')

    if args.mod_scheme == 'FSK':
        modulator = FskModulator(order=args.mod_order)
    elif args.mod_scheme == 'QAM':
        modulator = QamModulator(order=args.mod_order)

    modulated = modulator.modulate(encoded)

    print('=' * 72)
    mean_energies = modulator.compute_mean_energies(modulated)
    print(f'Mean energy per symbol : {mean_energies[0]:.4f} J/symbol')
    print(f'Mean energy per bit    : {mean_energies[1]:.4f} J/bit')

    if args.plot_constellation:
        plot_constellation(modulator.mapping)

    return 0


def plot_constellation(mapping):
    symbols = np.array(list(mapping.values()))

    if symbols.shape[1] != 2:
        raise ValueError('constellation must have 2 dimensions')

    x_min, x_max = np.min(symbols[:, 0]), np.max(symbols[:, 0])
    y_min, y_max = np.min(symbols[:, 1]), np.max(symbols[:, 1])
    x_len = x_max - x_min
    y_len = y_max - y_min
    xx, yy = np.meshgrid(
            np.linspace(x_min - 0.2*x_len, x_max + 0.2*x_len, 200),
            np.linspace(y_min - 0.2*y_len, y_max + 0.2*y_len, 200))

    # Extraer esta lógica a una función porque va a servir para
    # la demodulación también
    points = np.c_[xx.ravel(), yy.ravel()]
    distances = np.linalg.norm(points[:, None, :] - symbols, axis=-1)
    Z = np.argmin(distances, axis=1).reshape(xx.shape)

    cmap = 'tab20'
    plt.figure(figsize=(6, 6))
    plt.pcolormesh(xx, yy, Z, alpha=0.2, cmap=cmap)
    plt.contour(xx, yy, Z, linestyle='--', colors='black', levels=len(mapping))

    plt.scatter(symbols[:, 0], symbols[:, 1],
                c=list(int(k, 2) for k in mapping), cmap=cmap,
                edgecolors='black', label='Símbolos')

    for label, symbol in mapping.items():
        plt.text(symbol[0] + 0.02*x_len, symbol[1] + 0.02*y_len, repr(label))

    plt.legend()
    plt.show()


if __name__ == '__main__':
    import sys
    sys.exit(main())
