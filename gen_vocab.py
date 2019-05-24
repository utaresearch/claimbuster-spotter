import os
import pickle
from utils.vocab import get_vocab_information
from flags import FLAGS


def main():
    if os.path.isfile(FLAGS.vocab_loc):
        print("By running this script, you will be deleting all contents of " + FLAGS.vocab_loc)
        ans = input("Do you wish to continue? (y/n) ")
        if ans == 'y':
            print("Running code...")
            os.remove(FLAGS.vocab_loc)
        else:
            print("Exiting...")
            exit()

    print("Parsing vocab information...")

    with open(FLAGS.prc_data_loc, 'rb') as f:
        data = pickle.load(f)
    with open(FLAGS.vocab_loc, 'wb') as f:
        pickle.dump(get_vocab_information(data), f)

    print("Completed.")


if __name__ == '__main__':
    main()