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

    with open(FLAGS.prc_data_loc, 'rb') as f:
        data = pickle.load(f)
        all_vocab = get_vocab_information(data)

    if len(FLAGS.addition_vocab) > 0:
        print("Vocab files {} will also be sampled from".format(FLAGS.addition_vocab))

        for loc in FLAGS.addition_vocab:
            if loc == FLAGS.vocab_loc:
                continue
            with open(loc, 'rb') as f:
                data = pickle.load(f)
                np.concatenate((all_vocab, data))

    all_vocab = sorted(list(set(all_vocab)), key=lambda x: x[1], reverse=True)

    with open(FLAGS.vocab_loc, 'wb') as f:
        pickle.dump(all_vocab, f)

    print("Completed with {} vocabulary items.".format(len(all_vocab)))


if __name__ == '__main__':
    main()