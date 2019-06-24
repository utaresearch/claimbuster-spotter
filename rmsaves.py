import os
import argparse


extension_list = ['.data', '.meta', '.index']


def list_del(location):
    ret_ar = []
    directory = os.fsencode(location)
    for fstr in os.listdir(directory):
        if not args.clear_embeddings and 'embedding_matrix' in os.fsdecode(fstr):
            continue
        elif os.fsdecode(fstr) == 'checkpoint' or os.fsdecode(fstr) == 'tblogs':
            ret_ar.append(os.fsdecode(fstr))
        for ext in extension_list:
            if ext in os.fsdecode(fstr):
                ret_ar.append(os.fsdecode(fstr))
                break
    return ret_ar


def main():
    to_del = list_del(args.folder_to_clean)
    for el in to_del:
        os.remove(os.path.join(args.folder_to_clean, el))
    print('Process completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_to_clean', default='./output')
    parser.add_argument('--clear_embeddings', type=bool,  default=False)
    args = parser.parse_args()

    main()