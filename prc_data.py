import json, os, random
from data_utils import transformations as transf
import argparse
import pickle
import string
from pycontractions import Contractions
from tqdm import tqdm
from flags import FLAGS

cont = None
args = None
parser = None


def parse_json():
    with open(args.json_loc) as f:
        temp_data = json.load(f)
    dl = []

    for i in tqdm(range(len(temp_data)), ascii=True):
        f = temp_data[i]
        lab = f["label"]
        txt = list(cont.expand_texts([f["text"]], precise=True))[0]

        txt = txt.replace('-', ' ').lower().split(' ')
        if args.noun_rep:
            txt = transf.process_sentence_noun_rep(txt)
        elif args.full_tags:
            txt = transf.process_sentence_full_tags(txt)
        elif args.ner_spacy:
            txt = transf.process_sentence_ner_spacy(txt)

        words = txt.split(' ')
        for i in range(len(words)):
            words[i].strip(string.punctuation)

        txt = ' '.join(words)
        print(txt)

        dl.append((lab, txt))
    return dl


def parse_tags():
    global parser
    global args

    parser = argparse.ArgumentParser(description="Convert .json file to directory hierarchy and apply data transf.")
    parser.add_argument("--output_pkl", default="./output/prc_data.pickle")
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

    path = args.output_pkl[:args.output_pkl.find('/', args.output_pkl.find('/') + 1)]
    if not os.path.exists(path):
        os.mkdir(path)

    with open(args.output_pkl, 'wb') as f:
        pickle.dump(df, f)


def load_dependencies():
    global cont

    transf.load_dependencies(args)
    cont = Contractions(args.w2v_loc)

    print("Loading contractions model...")
    cont.load_models()
    print("Model loaded.")


def main():
    parse_tags()

    if os.path.isfile(args.output_pkl):
        print("By running this script, you will be deleting all contents of " + args.output_pkl)
        ans = input("Do you wish to continue? (y/n) ")
        if ans == 'y':
            print("Running code...")
            os.remove(args.output_pkl)
        else:
            print("Exiting...")
            exit()

    print("Loading dependencies...")
    load_dependencies()

    print("Processing data...")
    dl = parse_json()
    random.seed(FLAGS.random_state)
    random.shuffle(dl)

    write_pickle(dl)


if __name__ == "__main__":
    main()