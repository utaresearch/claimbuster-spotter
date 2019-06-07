import numpy as np
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

    all_vocab = set()

    with open(FLAGS.prc_data_loc, 'rb') as f:
        data = pickle.load(f)
        temp_voc = get_vocab_information(data)
        for word in temp_voc:
            all_vocab.add(word)

    if len(FLAGS.addition_vocab) > 0:
        for loc in FLAGS.addition_vocab:
            if loc == FLAGS.vocab_loc:
                continue
            print("Sampling from {}".format(loc))
            with open(loc, 'rb') as f:
                temp_voc = pickle.load(f)
                for word in temp_voc:
                    all_vocab.add(word)

    all_vocab = list(all_vocab)

    with open(FLAGS.vocab_loc, 'wb') as f:
        pickle.dump(all_vocab, f)

    print("Completed with {} vocabulary items.".format(len(all_vocab)))


if __name__ == '__main__':
    main()
