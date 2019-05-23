import argparse
from data_utils.vocab import get_vocab_information


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_pkl", default="./output/prc_data.pickle")
    args = parser.parse_args()

    print("Parsing vocab information...")
    print(get_vocab_information(args.output_pkl))


if __name__ == '__main__':
    main()