import pickle
import argparse
from data_utils.vocab import get_vocab_information


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./output/prc_data.pickle")
    parser.add_argument("--output", default="./output/vocab.pickle")
    args = parser.parse_args()

    print("Parsing vocab information...")

    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    with open(args.output, 'wb') as f:
        pickle.dump(get_vocab_information(data), f)

    print("Completed.")


if __name__ == '__main__':
    main()