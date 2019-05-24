import os
import argparse


extension_list = ['.data', '.meta', '.index']


def list_dirs(location):
    ret_ar = []
    directory = os.fsencode(location)
    for fstr in os.listdir(directory):
        if not args.clear_embeddings and 'embedding_matrix' in os.fsdecode(fstr):
            continue
        for ext in extension_list:
            if ext in os.fsdecode(fstr):
                ret_ar.append(os.fsdecode(fstr))
                break
    return ret_ar


def main():
    print(list_dirs(args.folder_to_clean))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_to_clean', default='./output')
    parser.add_argument('--clear_embeddings', type=bool,  default=False)
    args = parser.parse_args()

    main()