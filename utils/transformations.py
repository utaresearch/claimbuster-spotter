import nltk
import spacy
import string
import sys
import os
from textblob import TextBlob
from keras.preprocessing.text import text_to_word_sequence
from tqdm import tqdm

print(os.getcwd())
exit()

sys.path.append('..')
from flags import FLAGS

nlp = None
cont = None
embed_obj = None
kill_words = ["", "uh"]
stop_words = list(nltk.corpus.stopwords.words('english'))
pos_labels = list(nltk.load("help/tagsets/upenn_tagset.pickle").keys())

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
    "MIRVing": "attack with multiple independently targetable reentry vehicle",
    "Kardari": "Zardari",
    "countrypeople": "country people",
    "bicta": "",
    "bict": "",
    "l949": "1949",
    "l961": "1961",
    "198I": "1981",
    "undefensible": "indefensible",
    "198i": "1981",
    "Sholicatchvieli": "Shalikashvili",
    "ã¢â‚¬å“we": "we",
    "ã¢â‚¬â\x9d": "",
    "Chemomyrdin": "Chernomyrdin",
    "Chemomyrdin's": "Chernomyrdin",
    "revita1ize": "revitalize",
    "arterially": "from the arteries",
    "'80s": "1980",
    "'60s": "1960",
    "HMET": "heavy expanded mobility tactical truck",
    "hmett": "heavy expanded mobility tactical truck",
    "Vietnese": "Vietnamese",
    "namese": "",
    "''": "",
    "d'amato": "d'amato",
    "Shinsheki": "Shinseki",
    "exager": "exaggerated",
    "Cardash": "Radosh",
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
    "your$265": "your $265",
}

exp_contractions = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "here's": "here is",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "would",
    "I'd've": "I would have",
    "I'll": " will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'll": "that will",
    "that've": "that have",
    "that'd": "that had",
    "that'd've": "that would have",
    "that's": "that is",
    "'em": "them",
    "there're": "there are",
    "there've": "there have",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "there'll": "there will ",
    "they've": "they have",
    "to've": "to have",
    "'til": "until",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}


def expand_sentence(sentence):
    return [(strip_chars(word)) for word in sentence.split(' ')]


def correct_mistakes(sentence):
    sentence_list = expand_sentence(sentence)
    return ' '.join([pre + (dataset_specific_fixes[word] if word in dataset_specific_fixes else word) + post
                     for pre, word, post in sentence_list])


def expand_contractions(sentence):
    return ' '.join([(exp_contractions[word.lower()] if word.lower() in exp_contractions else word)
                     for word in sentence.split(' ')])


def remove_possessives(sentence):
    sentence = ' '.join(
        [(st if len(st) == 1 else (st[:-2] if st.rfind("'s") == len(st) - 2 else st)) for st in sentence.split(' ')])
    return ' '.join(
        [(st if len(st) == 1 else (st[:-2] if st.rfind("s'") == len(st) - 2 else st)) for st in sentence.split(' ')])


def remove_kill_words(sentence):
    ret = []
    for word in sentence.split(' '):
        if word not in kill_words:
            ret.append(word)
    return ' '.join(ret)


def strip_chars(inpstr, to_strip=string.punctuation):
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


def transform_sentence_complete(sentence):
    sentence = correct_mistakes(sentence)

    if not FLAGS.custom_preprc:
        return sentence

    sentence = (process_sentence_ner_spacy(sentence) if FLAGS.ner_spacy else sentence)
    sentence = ' '.join(text_to_word_sequence(sentence))

    sentence = expand_contractions(sentence)
    sentence = remove_possessives(sentence)
    sentence = remove_kill_words(sentence)

    return sentence


def process_dataset(inp_data):
    pos_tagged = []
    sentiments = []

    for i in tqdm(range(len(inp_data))):
        sentiments.append(get_sentiment(inp_data[i]))
        pos_tagged.append(process_sentence_full_tags(inp_data[i]))
        inp_data[i] = transform_sentence_complete(inp_data[i])

    return inp_data, pos_tagged, sentiments


def get_sentiment(inp_data):
    blob = TextBlob(inp_data)
    return [blob.polarity, blob.subjectivity]


###########################################################################

# ORIGINAL TRANSFORMATIONS

###########################################################################


def list_to_string(lst):
    ret = ""
    for f in lst:
        ret = ret + str(f) + " "
    return ret.rstrip()


def char_list_to_string(lst):
    ret = ""
    for f in lst:
        ret = ret + str(f)
    return ret


def get_tags(sentence):
    text = nltk.tokenize.word_tokenize(sentence)
    res = nltk.pos_tag(text)
    return res


def process_sentence_full_tags(sentence):
    prc_res = get_tags(sentence)
    ret = []
    for f in prc_res:
        ret.append(pos_labels.index(f[1]))
    return ret


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


def load_dependencies():
    global nlp, cont, embed_obj

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


def load_deps_dummy():
    global nlp

    print("Loading NLTK Dependencies...")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('tagsets')
    nltk.download('stopwords')
    # print("Loading Spacy NER Tagger...")
    # nlp = spacy.load("en_core_web_lg")
    # print("Tagger loaded.")
    print("NLTK dependencies Loaded.")


if __name__ == '__main__':
    load_deps_dummy()
    print(get_tags('I like %^*%(^(*%^#(^#^^#$#^#$^#^ $^4 7*$6 89$*(^ #*($^  $ $$$$$$$ $1343025823 million dollars'))
