import json, os, random
from utils import transformations as transf
import pickle
import string
from pycontractions import Contractions
from tqdm import tqdm
from flags import FLAGS

cont = None
kill_words = ["", "uh"]
random.seed(FLAGS.random_state)


def parse_json():
    with open(FLAGS.raw_data_loc) as f:
        temp_data = json.load(f)

    dl = []
    labels = [0, 0, 0]

    data_by_label = {
        -1: [],
        0: [],
        1: []
    }

    for i in tqdm(range(len(temp_data)), ascii=True):
        f = temp_data[i]
        lab = int(f["label"])
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

        data_by_label[lab].append(txt)

    if FLAGS.balance_NFS:
        random.shuffle(data_by_label[-1])
        data_by_label[-1] = data_by_label[-1][:len(data_by_label[1])]

    for key in data_by_label:
        for el in data_by_label[key]:
            dl.append((key, el))
            labels[int(key) + 1] += 1

    print(labels)
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
    path = FLAGS.prc_data_loc[:FLAGS.prc_data_loc.find('/', FLAGS.prc_data_loc.find('/') + 1)]
    if not os.path.exists(path):
        os.mkdir(path)

    with open(FLAGS.prc_data_loc, 'wb') as f:
        pickle.dump(df, f)


def load_dependencies():
    global cont

    transf.load_dependencies(FLAGS)
    cont = Contractions(FLAGS.w2v_loc)

    print("Loading contractions model...")
    cont.load_models()
    print("Model loaded.")


def main():
    parse_tags()

    if os.path.isfile(FLAGS.prc_data_loc):
        print("By running this script, you will be deleting all contents of " + FLAGS.prc_data_loc)
        ans = input("Do you wish to continue? (y/n) ")
        if ans == 'y':
            print("Running code...")
            os.remove(FLAGS.prc_data_loc)
        else:
            print("Exiting...")
            exit()

    print("Creating{}{} directory...".format(" missing " if not os.path.exists(FLAGS.output_dir)
                                             else " ", FLAGS.output_dir))
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    print("Loading dependencies...")
    load_dependencies()

    print("Processing data...")
    dl = parse_json()
    random.shuffle(dl)

    write_pickle(dl)


if __name__ == "__main__":
    main()