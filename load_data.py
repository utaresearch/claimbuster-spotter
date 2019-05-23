import json, os, random
from pycontractions import Contractions
from data_utils import transformations as transf
import argparse
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


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def set_dirs():
    global train_dir
    global test_dir
    global train_label_dir
    global test_label_dir
    create_dir(args.output_dir)
    train_dir = create_dir(args.output_dir + "/train")
    test_dir = create_dir(args.output_dir + "/test")
    train_label_dir.append(create_dir(train_dir + "/0"))
    train_label_dir.append(create_dir(train_dir + "/1"))
    train_label_dir.append(create_dir(train_dir + "/2"))
    test_label_dir.append(create_dir(test_dir + "/0"))
    test_label_dir.append(create_dir(test_dir + "/1"))
    test_label_dir.append(create_dir(test_dir + "/2"))


def split_into_dirs(dl):
    set_dirs()
    cutpoint = int(float(args.train_pct) / 100.0 * float(len(dl)))
    ctr = 0
    for i in range(0, len(dl)):  # training data
        sample = dl[i]
        lab = int(sample.label)
        s = sample.sentence

        if i <= cutpoint:
            target_dir = train_label_dir[lab]
        else:
            target_dir = test_label_dir[lab]
        with open(target_dir + "/" + str(ctr).zfill(5) + "_" + str(lab) + ".txt", "w") as f:
            f.write(s)

        ctr = ctr + 1


def print_sample_list(dl):
    for f in dl:
        print(f.label + " " + f.sentence)


def load_dependencies():
    global cont

    # Load NLTK deps
    if args.noun_rep or args.full_tags or args.ner_spacy:
        print("Loading NLTK Dependencies...")
        transf.nltk.download('punkt')
        transf.nltk.download('averaged_perceptron_tagger')
        transf.nltk.download('tagsets')
        if args.ner_stanford:
            print("Loading Stanford NER Tagger...")
            transf.st = transf.nltk.tag.StanfordNERTagger(args.ner_loc1, args.ner_loc2, encoding="utf-8")
            print("Tagger loaded.")
        if args.ner_spacy:
            print("Loading Spacy NER Tagger...")
            transf.nlp = transf.spacy.load("en_core_web_lg")
            print("Tagger loaded.")
        print("NLTK dependencies Loaded.")

    # Load word2vec model for contraction expansion
    print("Loading model from " + args.w2v_loc)
    cont = Contractions(args.w2v_loc)

    try:
        cont.load_models()
        print("Model Loaded.")
    except:
        raise Exception("Error: Model does not exist")


def parse_json():
    with open(args.json_loc) as f:
        temp_data = json.load(f)
    dl = []
    for i in tqdm(range(len(temp_data)), ascii=True):
        f = temp_data[i]
        if args.noun_rep:
            dl.append(
                Sample(str(int(f["label"]) + 1), transf.process_sentence_noun_rep(cont._expand_text(f["text"]))))
        elif args.full_tags:
            dl.append(
                Sample(str(int(f["label"]) + 1), transf.process_sentence_full_tags(cont._expand_text(f["text"]))))
        elif args.ner_stanford:
            dl.append(Sample(str(int(f["label"]) + 1),
                             transf.process_sentence_ner_stanford(args.ner_loc1, args.ner_loc2,
                                                                  cont._expand_text(f["text"]))))
        elif args.ner_spacy:
            dl.append(
                Sample(str(int(f["label"]) + 1), transf.process_sentence_ner_spacy(cont._expand_text(f["text"]))))
        else:
            dl.append(Sample(str(int(f["label"]) + 1), cont._expand_text(f["text"])))
    return dl


def parse_tags():
    global parser
    global args

    parser = argparse.ArgumentParser(description="Convert .json file to directory hierarchy and apply data transf.")
    parser.add_argument("--output_dir", default="./output/prc_data")
    parser.add_argument("--json_loc", default="./data/data_large.json")
    parser.add_argument("--w2v_loc", default="./data/word2vec/GoogleNews-vectors-negative300.bin")
    parser.add_argument("--train_pct", type=int, default=75)
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


def main():
    parse_tags()
    if os.path.exists(args.output_dir):
        print("By running this file, you will be deleting all contents of " + args.output_dir)
        ans = input("Do you wish to continue? (y/n)")
        if ans == 'y':
            print("Running code...")
            os.system("rm -r " + args.output_dir)
        else:
            print("Exiting...")
            exit()

    load_dependencies()
    print("Processing data...")
    dl = parse_json()
    random.seed(456)
    random.shuffle(dl)
    print("Creating directories...")
    split_into_dirs(dl)
    print("Processing complete.")


if __name__ == "__main__":
    main()