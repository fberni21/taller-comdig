import argparse
import string

import source_encoder as se

from utils import compute_char_probs, compute_entropy


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

    huff = se.HuffmanEncoder(probs)

    print(f'Minimum length   : {entropy:.4f} bits/symbol')
    print(f'Code mean length : {huff.length_avg:.4f} bits/symbol')
    print(f'Code efficiency  : {entropy/huff.length_avg:.4f}')
    print(f'Code variance    : {huff.length_var:.4f} (bits/symbol)^2')

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

    source_enc = se.SourceEncoder(huff.code_map)
    encoded = source_enc.encode(chars)

    decode_map = {v: chr(k) for k, v in enumerate(huff.code_map)}
    decoded = decode(encoded, decode_map)

    with open('decoded.txt', 'w') as f:
        f.write(decoded)

    return 0


def decode(encoded, decode_map):
    length = len(encoded)
    decoded = []
    start = 0
    while start < length:
        end = start + 1
        while end < length and encoded[start:end] not in decode_map:
            end += 1
        decoded.append(decode_map[encoded[start:end]])
        start = end
    return ''.join(decoded)


if __name__ == '__main__':
    import sys
    sys.exit(main())
