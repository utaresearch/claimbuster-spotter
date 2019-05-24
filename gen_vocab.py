import os
import pickle
import argparse
from utils.vocab import get_vocab_information


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./output/prc_data.pickle")
    parser.add_argument("--output", default="./output/vocab.pickle")
    args = parser.parse_args()

    if os.path.isfile(args.output):
        print("By running this script, you will be deleting all contents of " + args.output)
        ans = input("Do you wish to continue? (y/n) ")
        if ans == 'y':
            print("Running code...")
            os.remove(args.output)
        else:
            print("Exiting...")
            exit()

    print("Parsing vocab information...")

    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    with open(args.output, 'wb') as f:
        pickle.dump(get_vocab_information(data), f)

    print("Completed.")


if __name__ == '__main__':
    main()