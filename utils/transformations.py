import nltk
import spacy
import string
import sys
sys.path.append('..')
from flags import FLAGS
from pycontractions import Contractions

nlp = None
cont = None
kill_words = ["", "uh"]
spacy_to_nl = {
    "PERSON": "person",
    "NORP": "nationality",
    "FAC": "infrastructure",
    "ORG": "organization",
    "GPE": "country",
    "LOC": "location",
    "PRODUCT": "product",
    "EVENT": "event",
    "WORK_OF_ART": "art",
    "LAW": "law",
    "LANGUAGE": "language",
    "DATE": "date",
    "TIME": "time",
    "PERCENT": "percentage",
    "MONEY": "money",
    "QUANTITY": "quantity",
    "ORDINAL": "first",
    "CARDINAL": "number"
}


def list_to_string(list):
    ret = ""
    for f in list:
        ret = ret + f + " "
    return ret.rstrip()


def char_list_to_string(list):
    ret = ""
    for f in list:
        ret = ret + f
    return ret


def get_tags(sentence):
    text = nltk.tokenize.word_tokenize(sentence)
    res = nltk.pos_tag(text)
    return res


def process_sentence_noun_rep(sentence):
    prc_res = get_tags(sentence)
    ret = []
    for p in prc_res:
        if p[1] == "POS":
            continue
        elif "NN" in p[1]:
            ret.append("noun")
        else:
            ret.append(p[0])
    return list_to_string(ret)


def process_sentence_full_tags(sentence):
    prc_res = get_tags(sentence)
    ret = []
    for f in prc_res:
        ret.append(f[1])
    return list_to_string(ret)


def process_sentence_ner_spacy(sentence):
    doc = nlp(sentence)
    ret = list(sentence)
    adj = 0
    for ent in doc.ents:
        newlab = spacy_to_nl[ent.label_]
        del ret[ent.start_char - adj:ent.end_char - adj]
        temp = list(newlab)
        ret[ent.start_char - adj:ent.start_char - adj] = temp
        adj = adj + len(ent.text) - len(newlab)
    return char_list_to_string(ret)


def transform_sentence_complete(sentence):
    txt = list(cont.expand_texts([sentence], precise=True))[0]
    txt = txt.replace('-', ' ').lower()
    if FLAGS.noun_rep:
        txt = process_sentence_noun_rep(txt)
    elif FLAGS.full_tags:
        txt = process_sentence_full_tags(txt)
    elif FLAGS.ner_spacy:
        txt = process_sentence_ner_spacy(txt)

    words = txt.split(' ')
    for j in range(len(words)):
        words[j] = words[j].strip(string.punctuation)
        if words[j].isdigit():
            words[j] = "NUM"

    txt = []
    for word in words:
        if word not in kill_words:
            txt.append(word)

    return ' '.join(txt)


def load_dependencies():
    global nlp, cont

    # Load NLTK deps
    if FLAGS.noun_rep or FLAGS.full_tags or FLAGS.ner_spacy:
        print("Loading NLTK Dependencies...")
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('tagsets')
        if FLAGS.ner_spacy:
            print("Loading Spacy NER Tagger...")
            nlp = spacy.load("en_core_web_lg")
            print("Tagger loaded.")
        print("NLTK dependencies Loaded.")

    # Load word2vec model for contraction expansion
    print("Loading model from " + FLAGS.w2v_loc)
    cont = Contractions(FLAGS.w2v_loc)

    try:
        cont.load_models()
        print("Model Loaded.")
    except:
        raise Exception("Error: Model does not exist")


def load_deps_dummy():
    global nlp

    print("Loading NLTK Dependencies...")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('tagsets')
    print("Loading Spacy NER Tagger...")
    nlp = spacy.load("en_core_web_lg")
    print("Tagger loaded.")
    print("NLTK dependencies Loaded.")


if __name__ == '__main__':
    load_deps_dummy()
    print(process_sentence_ner_spacy('I like Kevin Meng because he has lots of $50 million.'))
