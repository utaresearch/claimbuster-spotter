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
stop_words = list(nltk.corpus.stopwords.words('english'))
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

dataset_specific_fixes = {
    "itã\x8fâ‚¬s": "it is",
    "itÃ¢â‚¬â„¢s": "it is",
    "Midgetman": "",
    "naãƒâ¯ve": "naive",
    "1990ã\x8fâ‚¬s": "year 1990",
    "30ã\x8fâ‚¬s": "1930",
    "40ã\x8fâ‚¬s": "1940",
    "'40's": "1940",
    "'50's": "1950",
    "'60's": "1960",
    "'87": "1987",
    "'81": "1981",
    "'77": "1977",
    "'83": "1983",
    "'94": "1994",
    "'93": "1993",
    "'97": "1997",
    "'92": "1992",
    "ã¢â€°â¤": "",
    "ã¢â€°â¥mr": "",
    "Ã¢â€°Â¤": "",
    "ã¢â€°â¥who": "who",
    "aayuh": "",
    "mirving": "attack with multiple independently targetable reentry vehicle",
    "kardari": "zardari",
    "countrypeople": "country people",
    "bicta": "",
    "bict": "",
    "l949": "1949",
    "l961": "1961",
    "198I": "1981",
    "undefensible": "indefensible",
    "198i": "1981",
    "sholicatchvieli": "shalikashvili",
    "ã¢â‚¬å“we": "we",
    "ã¢â‚¬â\x9d": "",
    "chemomyrdin": "chernomyrdin",
    "chemomyrdin's": "chernomyrdin",
    "revita1ize": "revitalize",
    "arterially": "from the arteries",
    "'80s": "1980",
    "'60s": "1960",
    "hmet": "heavy expanded mobility tactical truck",
    "hmett": "heavy expanded mobility tactical truck",
    "Vietnese": "Vietnamese",
    "namese": "",
    "''": "",
    "d'amato": "d'amato",
    "Shinsheki": "Shinseki",
    "exager": "exaggerated",
    "cardash": "radosh",
    "youã¢â‚¬â„¢re": "you are",
    "treasurey": "treasury",
    "itã¢â‚¬â„¢s": "it is",
    "iã¢â‚¬â„¢ll": "i will",
    "ã‚": "",
    "weã¢â‚¬â„¢ll": "we will",
    "ãƒâ¢ã¢â€šâ¬ã¢â‚¬å“": "",
    "270billion": "270 billion",
    "youã¢â‚¬â„¢ve": "you have",
    "G.N.P": "gross national product",
    "STARTer": "starter",
    "ask/do": "ask or do",
    "K.G.B": "security service",
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
    txt = txt.replace('-', ' ')

    def strip_chars(inpstr, to_strip):
        strar = list(inpstr)
        stripped_away_front = ""
        stripped_away_back = ""

        for idx in reversed(range(0, len(strar))):
            if strar[idx] in to_strip:
                stripped_away_back += strar[idx]
                del strar[idx]
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
            lcount += 1

        return stripped_away_front, ''.join(strar), stripped_away_back[::-1]

    def remove_possessive(st):
        return st if len(st) == 1 else (st[:-2] if st.rfind("'s") == len(st) - 2 else st)

    txt_split = txt.split(' ')
    changed_words = []
    for i in range(len(txt_split)):
        temp_word = txt_split[i]
        if temp_word in dataset_specific_fixes:
            changed_words.append(temp_word)
            txt_split[i] = dataset_specific_fixes[temp_word]
        else:
            temp_pre, temp_word, temp_post = strip_chars(txt_split[i], string.punctuation)
            if temp_word in dataset_specific_fixes:
                changed_words.append(temp_word)
                txt_split[i] = temp_pre + dataset_specific_fixes[temp_word] + temp_post
    txt = ' '.join((' '.join(txt_split)).split(' '))

    if FLAGS.noun_rep:
        txt = process_sentence_noun_rep(txt)
    elif FLAGS.full_tags:
        txt = process_sentence_full_tags(txt)
    elif FLAGS.ner_spacy:
        txt = process_sentence_ner_spacy(txt)

    words = txt.split(' ')
    ret_words = []
    for j in range(len(words)):
        str_front, new_word, str_back = strip_chars(words[j], string.punctuation)

        if str_front not in kill_words:
            for ch in str_front:
                ret_words.append(ch)
        if new_word not in kill_words and (not FLAGS.remove_stopwords or new_word not in stop_words):
            ret_words.append(remove_possessive(new_word))
        if str_back not in kill_words:
            for ch in str_back:
                ret_words.append(ch)

    return ' '.join(ret_words), changed_words


def load_dependencies():
    global nlp, cont, embed_obj

    # Load NLTK deps
    if FLAGS.noun_rep or FLAGS.full_tags or FLAGS.ner_spacy:
        print("Loading NLTK Dependencies...")
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('tagsets')
        nltk.download('stopwords')
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
    print(process_sentence_ner_spacy('My name is Daniel and I like to eat apples at Walmart.'))
