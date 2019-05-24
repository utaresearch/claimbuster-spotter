import json, os, random
from utils import transformations as transf
import pickle
import string
from pycontractions import Contractions
from tqdm import tqdm
from flags import FLAGS

cont = None
args = None
parser = None
kill_words = ["", "uh"]


def parse_json():
    with open(args.json_loc) as f:
        temp_data = json.load(f)
    dl = []

    for i in tqdm(range(len(temp_data)), ascii=True):
        f = temp_data[i]
        lab = f["label"]
        txt = list(cont.expand_texts([f["text"]], precise=True))[0]

        txt = txt.replace('-', ' ').lower()
        if FLAGS.noun_rep:
            txt = transf.process_sentence_noun_rep(txt)
        elif FLAGS.full_tags:
            txt = transf.process_sentence_full_tags(txt)
        elif FLAGS.ner_spacy:
            txt = transf.process_sentence_ner_spacy(txt)

        words = txt.split(' ')
        for j in range(len(words)):
            words[j] = words[j].strip(string.punctuation)
            if words[j].isdigit():
                words[j] = "NUM"

        txt = []
        for word in words:
            if word not in kill_words:
                txt.append(word)

        txt = ' '.join(txt)

        dl.append((lab, txt))
    return dl


def parse_tags():
    cnt = 0
    if FLAGS.noun_rep:
        cnt += 1
    if FLAGS.full_tags:
        cnt += 1
    if FLAGS.ner_spacy:
        cnt += 1
    if cnt > 1:
        raise Exception("You cannot have more than one data transformation option to be True at once.")


def write_pickle(df):
    global args

    path = os.path.join(FLAGS.output_dir, FLAGS.prc_data_loc)[:os.path.join(FLAGS.output_dir, FLAGS.prc_data_loc).find('/', os.path.join(FLAGS.output_dir, FLAGS.prc_data_loc).find('/') + 1)]
    if not os.path.exists(path):
        os.mkdir(path)

    with open(os.path.join(FLAGS.output_dir, FLAGS.prc_data_loc), 'wb') as f:
        pickle.dump(df, f)


def load_dependencies():
    global cont

    transf.load_dependencies(args)
    cont = Contractions(FLAGS.w2v_loc)

    print("Loading contractions model...")
    cont.load_models()
    print("Model loaded.")


def main():
    parse_tags()

    if os.path.isfile(os.path.join(FLAGS.output_dir, FLAGS.prc_data_loc)):
        print("By running this script, you will be deleting all contents of " + os.path.join(FLAGS.output_dir, FLAGS.prc_data_loc))
        ans = input("Do you wish to continue? (y/n) ")
        if ans == 'y':
            print("Running code...")
            os.remove(os.path.join(FLAGS.output_dir, FLAGS.prc_data_loc))
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