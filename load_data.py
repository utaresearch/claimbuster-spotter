import json, os, random
from data_utils import transformations as transf
import argparse
import pickle
from pycontractions import Contractions
from tqdm import tqdm

train_dir = ""
test_dir = ""
train_label_dir = []
test_label_dir = []
cont = None
args = None
parser = None


class Sample:
    label = ""
    sentence = ""

    def __init__(self, l, s):
        self.label = l
        self.sentence = s


def parse_json():
    with open(args.json_loc) as f:
        temp_data = json.load(f)
    dl = []

    load_dependencies()
    for i in tqdm(range(len(temp_data)), ascii=True):
        f = temp_data[i]
        lab = f["label"]
        txt = list(cont.expand_texts([f["text"]], precise=True))[0]
        print(lab)
        print(f["text"])
        print(txt)
        if args.noun_rep:
            dl.append(Sample(lab, transf.process_sentence_noun_rep(txt)))
        elif args.full_tags:
            dl.append(Sample(lab, transf.process_sentence_full_tags(txt)))
        elif args.ner_spacy:
            dl.append(Sample(lab, transf.process_sentence_ner_spacy(txt)))
        else:
            dl.append(Sample(lab, txt))
    return dl


def parse_tags():
    global parser
    global args

    parser = argparse.ArgumentParser(description="Convert .json file to directory hierarchy and apply data transf.")
    parser.add_argument("--output_pkl", default="./output/prc_data.pkl")
    parser.add_argument("--json_loc", default="./data/data_small.json")
    parser.add_argument("--w2v_loc", default="./data/word2vec/GoogleNews-vectors-negative300.bin")
    parser.add_argument("--noun_rep", type=bool, default=False)
    parser.add_argument("--full_tags", type=bool, default=False)
    parser.add_argument("--ner_spacy", type=bool, default=True)
    args = parser.parse_args()

    cnt = 0
    if args.noun_rep:
        cnt += 1
    if args.full_tags:
        cnt += 1
    if args.ner_spacy:
        cnt += 1
    if cnt > 1:
        raise Exception("You cannot have more than one data transformation option to be True at once.")


def write_pickle(df):
    global args

    with open(args.output_pkl, 'wb') as f:
        pickle.dump(df, f)


def load_dependencies():
    global cont

    cont = Contractions(args.w2v_loc)
    transf.load_dependencies(args)


def main():
    parse_tags()

    if os.path.isfile(args.output_pkl):
        print("By running this script, you will be deleting all contents of " + args.output_pkl)
        ans = input("Do you wish to continue? (y/n)")
        if ans == 'y':
            print("Running code...")
            os.remove(args.output_pkl)
        else:
            print("Exiting...")
            exit()

    print("Processing data...")

    dl = parse_json()
    random.seed(456)
    random.shuffle(dl)

    print(dl)
    write_pickle(dl)


if __name__ == "__main__":
    main()