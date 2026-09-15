import argparse
import string

import source

from utils import compute_char_probs, compute_entropy, is_prefix_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='Input file name')
    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        text = file.read()

    chars = [ord(c) for c in text]

    probs = compute_char_probs(chars)
    entropy = compute_entropy(probs)
    print(f'File entropy     : {entropy:.4f} bits/char')

    huff = source.HuffmanEncoder(probs)

    print(f'Minimum length   : {entropy:.4f} bits/symbol')
    print(f'Code mean length : {huff.length_avg:.4f} bits/symbol')
    print(f'Code efficiency  : {entropy/huff.length_avg:.4f}')
    print(f'Code variance    : {huff.length_var:.4f} (bits/symbol)^2')

    print(f'Code is{''
          if is_prefix_code(huff.code_map) else 'not '} a prefix code')

    for k, v in sorted(enumerate(huff.code_map),
                       key=lambda it: probs[it[0]],
                       reverse=True):
        p = probs[k]
        if p == 0:
            continue
        c = f'`{chr(k)}`'
        if chr(k) not in string.ascii_letters + ' ' + string.digits \
                + string.punctuation:
            c = f'0x{k:02x}'
        print(f'{c}\t{p:.8f}\t{v}')

    source_enc = source.SourceEncoder(huff.code_map)
    encoded = source_enc.encode(chars)

    source_dec = source.SourceDecoder(huff.code_map)
    decoded = source_dec.decode(encoded)

    with open('decoded.txt', 'w') as f:
        f.write(decoded)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
