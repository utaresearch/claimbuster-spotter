import nltk
import spacy
import string
import sys
sys.path.append('..')
from flags import FLAGS
from pycontractions import Contractions

nlp = None
cont = None
embed_obj = None
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
    # txt = txt.replace('-', ' ').lower()
    txt = txt.lower()

    if FLAGS.noun_rep:
        txt = process_sentence_noun_rep(txt)
    elif FLAGS.full_tags:
        txt = process_sentence_full_tags(txt)
    elif FLAGS.ner_spacy:
        txt = process_sentence_ner_spacy(txt)

    def strip_chars(inpstr, to_strip):
        strar = list(inpstr)
        stripped_away_front = ""
        stripped_away_back = ""

        for i in reversed(range(0, len(strar))):
            if strar[i] in to_strip:
                stripped_away_back += strar[i]
                del strar[i]
            else:
                break
        lcount = 0
        while lcount < len(strar):
            if strar[lcount] in to_strip:
                stripped_away_front += strar[lcount]
                del strar[lcount]
                lcount -= 1
            else:
                break

        return stripped_away_front, ''.join(strar), stripped_away_back[::-1]

    words = txt.split(' ')
    ret_words = []
    for j in range(len(words)):
        str_front, new_word, str_back = strip_chars(words[j], string.punctuation)

        if str_front not in kill_words:
            ret_words.append(str_front)
        if new_word not in kill_words:
            ret_words.append(new_word)
        if str_back not in kill_words:
            ret_words.append(str_back)

    return ' '.join(ret_words)


def load_dependencies():
    global nlp, cont, embed_obj

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
    print("Loading model from " + FLAGS.w2v_loc_bin)
    cont = Contractions(FLAGS.w2v_loc_bin)

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
    print(process_sentence_ner_spacy('I like Donald Trump because he has lots of $50 million.'))
