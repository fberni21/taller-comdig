import argparse

import source

from utils import compute_char_probs, compute_entropy, is_prefix_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='input file name')
    parser.add_argument('--sample', default=None,
                        help='sample line to encode and decode')
    parser.add_argument('--output', default='decoded.txt',
                        help='decoded file name (default: decoded.txt)')
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
    for k in sorted(range(128),
                    key=lambda k: probs[k],
                    reverse=True):
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

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
