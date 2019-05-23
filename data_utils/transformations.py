import nltk
import spacy
st = None
nlp = None


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


def process_sentence_ner_stanford(tag1, tag2, sentence):
    tokenized_text = nltk.tokenize.word_tokenize(sentence)
    classified_text = st.tag(tokenized_text)
    ret = []
    for f in classified_text:
        if f[1] == 'O':
            ret.append(f[0])
        else:
            ret.append(f[1])
    return list_to_string(ret)


def process_sentence_ner_spacy(sentence):
    doc = nlp(sentence)
    ret = list(sentence)
    adj = 0
    for ent in doc.ents:
        del ret[ent.start_char - adj:ent.end_char - adj]
        temp = list(ent.label_)
        ret[ent.start_char - adj:ent.start_char - adj] = temp
        adj = adj + len(ent.text) - len(ent.label_)
    return char_list_to_string(ret)
