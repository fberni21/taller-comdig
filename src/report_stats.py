"""
Genera los datos estadisticos necesarios para completar el informe de la
Seccion B (Codificacion y Decodificacion de Fuente): tabla completa de
caracteres, tabla resumen de entropia/longitudes/eficiencia, comparacion de
bits totales, y ejemplo de codificacion/decodificacion de una linea de texto.

Uso:
    python report_stats.py <archivo_texto> [--sample "linea de ejemplo"]

Si no se pasa --sample, se toma la primera linea del archivo de mas de
30 caracteres como ejemplo.
"""
import argparse
import string
import sys

import source
from utils import compute_char_probs, compute_entropy, is_prefix_code


def char_repr(k):
    c = chr(k)
    if c in string.ascii_letters + ' ' + string.digits + string.punctuation:
        return c
    if c == '\n':
        return '\\n'
    return f'0x{k:02x}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='Archivo de texto a analizar')
    parser.add_argument('--sample', default=None,
                         help='Linea de texto a codificar/decodificar como ejemplo (item d)')
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        text = f.read()

    chars = [ord(c) for c in text]
    total_chars = len(chars)

    probs = compute_char_probs(chars)
    entropy = compute_entropy(probs)
    huff = source.HuffmanEncoder(probs)
    enc = source.SourceEncoder(huff.code_map)
    dec = source.SourceDecoder(huff.code_map)

    print('=' * 60)
    print('TABLA a) -- caracteres, cantidad, probabilidad y codigo')
    print('=' * 60)
    print(f'{"char":6}{"cantidad":>12}{"prob":>14}{"codigo":>25}')
    order = sorted(range(256), key=lambda k: probs[k], reverse=True)
    for k in order:
        p = probs[k]
        if p == 0:
            continue
        count = round(p * total_chars)
        print(f'{char_repr(k):6}{count:12d}{p:14.8f}{huff.code_map[k]:>25}')

    print()
    print('=' * 60)
    print('TABLA e) -- resumen entropia / longitudes / eficiencia')
    print('=' * 60)
    print(f'Entropia de la fuente         : {entropy:.4f} bits/simbolo')
    print(f'Longitud minima (Shannon)     : {entropy:.4f} bits/simbolo')
    print(f'Longitud media del codigo     : {huff.length_avg:.4f} bits/simbolo')
    print(f'Varianza del codigo           : {huff.length_var:.4f} (bits/simbolo)^2')
    print(f'Eficiencia                    : {entropy/huff.length_avg:.4f}')
    print(f'Longitud codigo longitud fija : 8 bits/simbolo (ASCII extendido)')
    print(f'Codigo es prefijo             : {is_prefix_code(huff.code_map)}')

    print()
    print('=' * 60)
    print('TABLA f) -- bits totales necesarios')
    print('=' * 60)
    total_bits_designed = sum(len(huff.code_map[c]) for c in chars)
    total_bits_fixed = total_chars * 8
    ahorro = 100 * (1 - total_bits_designed / total_bits_fixed)
    print(f'Cantidad total de caracteres       : {total_chars}')
    print(f'Bits totales (codigo disenado)     : {total_bits_designed}')
    print(f'Bits totales (codigo longitud fija): {total_bits_fixed}')
    print(f'Ahorro                             : {ahorro:.2f}%')

    print()
    print('=' * 60)
    print('ITEM d) -- ejemplo de codificacion/decodificacion')
    print('=' * 60)
    sample = args.sample
    if sample is None:
        for line in text.split('\n'):
            if 30 < len(line) < 60:
                sample = line
                break
    sample_chars = [ord(c) for c in sample]
    encoded = enc.encode(sample_chars)
    decoded = dec.decode(encoded)
    print(f'Original     : {sample!r}')
    print(f'Codificada   : {encoded}')
    print(f'Bits         : {len(encoded)}')
    print(f'Decodificada : {decoded!r}')
    print(f'Coincide     : {sample == decoded}')

    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: ts=4 sts=4 sw=4 et lbr
