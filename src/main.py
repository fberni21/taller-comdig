import argparse
import source_encoder as se
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='Input file name')
    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        text = file.read()

    probs = se.compute_char_probs(text)
    entropy = se.compute_entropy(probs)
    print(f'File entropy:\t\t{entropy:.4f} bits/char')

    code_map = se.huff_encode(probs)
    code_lengths = np.asarray([len(code_map[key]) for key in range(256)])
    code_len_mean = np.sum(probs * code_lengths)
    code_len_var = np.sum(probs * (code_lengths - code_len_mean)**2)

    print(f'Minimum length:\t\t{entropy:.4f} bits/symbol')
    print(f'Code mean length:\t{code_len_mean:.4f} bits/symbol')
    print(f'Code efficiency:\t{entropy/code_len_mean:.4f}')
    print(f'Code variance:\t\t{code_len_var:.4f} (bits/symbol)^2')

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
