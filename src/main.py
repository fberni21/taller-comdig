import argparse
import source_encoder as se


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('filename', help='Input file name')
        args = parser.parse_args()

        print(se.compute_file_stats(args.filename))
    except Exception as e:
        print(f'Error: {e}')
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
