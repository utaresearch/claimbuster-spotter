import tensorflow as tf
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import f1_score
import math
import nltk
import spacy
import string
from textblob import TextBlob
from keras.preprocessing.text import text_to_word_sequence
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from keras.preprocessing.text import Tokenizer
from sklearn.utils import resample
import shutil
import json
import pickle
from tqdm import tqdm


flags = tf.flags
FLAGS = flags.FLAGS


# ------------------------- FLAGS FOR 2-CLASS TRAINING -------------------------

# Re-Copy later when needed

# ------------------------- FLAGS FOR 3-CLASS TRAINING -------------------------

# Hardware
flags.DEFINE_list('gpu_active', [0], 'ID of GPU to use: in range [0, 4]')

# Preprocessing
flags.DEFINE_bool('ner_spacy', False, 'Named entity recognition with spaCy')

# Base directories
flags.DEFINE_string('output_dir', './model', 'Location of outputs')
flags.DEFINE_string('data_dir', './data', 'Location of data')

# Data
flags.DEFINE_bool('refresh_data', False, 'Re-process ./data/all_data.pickle')
flags.DEFINE_integer('max_len', 200, 'Maximum length of input')
flags.DEFINE_bool('remove_stopwords', False, 'Remove stop words (e.g. the, a, etc.)')
flags.DEFINE_bool('sklearn_oversample', True, 'Oversample underrepresented classes with sklearn')
flags.DEFINE_bool('weight_classes_loss', False, 'Weight classes in CE loss function')
flags.DEFINE_list('addition_vocab', ['./data/disjoint_2000/vocab.pickle'], 'Additional corpuses to sample vocab data from')

# Eval
flags.DEFINE_integer('stat_print_interval', 1, 'Numbers of epochs before stats are printed again')
flags.DEFINE_integer('model_save_interval', 1, 'Numbers of epochs before model is saved again')

# Data v2
flags.DEFINE_integer('total_examples', None, 'Total number of examples')
flags.DEFINE_integer('train_examples', None, 'Number of training examples')
flags.DEFINE_integer('test_examples', None, 'Number of testing examples')
flags.DEFINE_integer('random_state', 59, 'State of pseudo-randomness')

# Model architecture
flags.DEFINE_bool('bert_model', False, 'Use BERT pretrained RNN for NL LSTM.')
flags.DEFINE_integer('rnn_num_layers', 1, 'Number of LSTM layers.')
flags.DEFINE_integer('rnn_cell_size', 16, 'Number of hidden units in the LSTM.')
flags.DEFINE_bool('bidir_lstm', True, 'Use bidirectional LSTM')

# Optimization
flags.DEFINE_integer('max_steps', 1000, 'Number of epochs to run.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate while during optimiation.')

# Regularization
flags.DEFINE_float('l2_reg_coeff', 0.001, 'If val > 0, use L2 Regularization on weights in graph')
flags.DEFINE_float('keep_prob_cls', 0.6, 'Keep probability of classification layer.')
flags.DEFINE_float('keep_prob_lstm', 0.6, 'Keep probability LSTM network.')

# Embeddings
flags.DEFINE_bool('elmo_embed', False, 'dummy flag')
flags.DEFINE_integer('embed_type', 1, '0 for word2vec, 1 for Stanford glove')
flags.DEFINE_string('w2v_loc', './data/word2vec/w2v3b_gensim.txt', 'Location of w2v embeddings')
# flags.DEFINE_string('glove_loc', './data/glove/glove840b_gensim.txt', 'Location of glove embeddings')
flags.DEFINE_string('glove_loc', './data/glove/glove6b100d_gensim.txt', 'Location of glove embeddings')
flags.DEFINE_string('w2v_loc_bin', './data/word2vec/w2v3b_gensim.bin', 'Location of w2v embeddings in BINARY form')
flags.DEFINE_bool('train_embed', False, 'Train on top of w2v embeddings')  # we don't have enough data to train embed
flags.DEFINE_integer('embedding_dims', 100, 'Dimensions of embedded vector.')
flags.DEFINE_bool('random_init_oov', False, 'Use np.random.normal init for unknown embeddings. 0-fill if False')

# Adversarial and virtual adversarial training parameters.
flags.DEFINE_bool('adv_train', False, 'Train using adversarial perturbations')
flags.DEFINE_float('adv_coeff', 1.0, 'Coefficient of adversarial loss')
flags.DEFINE_float('perturb_norm_length', 6.0, 'Norm length of adversarial perturbation')

# Output stats
flags.DEFINE_integer('num_classes', 3, 'Number of classes for classification (2 combines NFS and UFS)')

# Training
flags.DEFINE_bool('adam', True, 'Adam or RMSProp if False')
flags.DEFINE_bool('restore_and_continue', False, 'Restore previous training session and continue')
flags.DEFINE_integer('batch_size', 128, 'Size of the batch.')

# Locations (must be last due to customization)
flags.DEFINE_string('raw_data_loc', '{}/data_small.json'.format(FLAGS.data_dir), 'Location of raw data')
flags.DEFINE_string('raw_dj_eval_loc', '{}/disjoint_2000.json'.format(FLAGS.data_dir), 'Location of raw data')
flags.DEFINE_string('prc_data_loc', '{}/all_data.pickle'.format(FLAGS.data_dir), 'Location of saved processed data')

if not os.path.isfile(FLAGS.prc_data_loc):
    FLAGS.refresh_data = True


def print_flags():
    tf.logging.info(FLAGS.flag_values_dict())


class Dataset:
    x = []
    y = []

    def __init__(self, x, y, random_state):
        self.x = x
        self.y = y

        self.random_state = random_state
        self.shuffle()

    def shuffle(self):
        self.x, self.y = shuffle(self.x, self.y, random_state=self.random_state)

    def get_length(self):
        xlen, ylen = len(self.x), len(self.y)
        if xlen != ylen:
            raise ValueError("size of x != size of y ({} != {})".format(xlen, ylen))
        return xlen


class DataLoader:
    def __init__(self):
        assert FLAGS.num_classes == 2 or FLAGS.num_classes == 3

        self.data, self.eval_data, self.vocab = self.load_external_raw()

        if FLAGS.num_classes == 2:
            self.convert_3_to_2()

        self.class_weights = self.compute_class_weights()
        tf.logging.info('Class weights computed to be {}'.format(self.class_weights))

        self.data.shuffle()
        self.post_process_flags()

    def convert_3_to_2(self):
        self.data.y = [(1 if self.data.y[i] == 2 else 0) for i in range(len(self.data.y))]
        self.eval_data.y = [(1 if self.eval_data.y[i] == 2 else 0) for i in range(len(self.eval_data.y))]

    def compute_class_weights(self):
        ret = compute_class_weight('balanced', [z for z in range(FLAGS.num_classes)], self.data.y)

        if FLAGS.num_classes == 3:
            ret[1] /= 4

        return ret

    def load_training_data(self):
        ret = Dataset(self.data.x, self.data.y, FLAGS.random_state)

        if FLAGS.sklearn_oversample:
            classes = [[] for _ in range(FLAGS.num_classes)]

            for i in range(len(ret.x)):
                classes[ret.y[i]].append(ret.x[i])

            if FLAGS.num_classes == 3:
                maj_len = len(classes[2])
                classes[0] = resample(classes[0], n_samples=int(maj_len * 2.75), random_state=FLAGS.random_state)
                classes[1] = resample(classes[1], n_samples=int(maj_len * 0.90), random_state=FLAGS.random_state)
                classes[2] = resample(classes[2], n_samples=int(maj_len * 1.50), random_state=FLAGS.random_state)
            else:
                pass
                # maj_len = len(classes[0])
                # # classes[0] = resample(classes[0], n_samples=int(maj_len), random_state=FLAGS.random_state)
                # classes[1] = resample(classes[1], n_samples=int(maj_len * 0.40), random_state=FLAGS.random_state)

            ret = Dataset([], [], random_state=FLAGS.random_state)
            del self.data.x[:FLAGS.train_examples]
            del self.data.y[:FLAGS.train_examples]

            for lab in range(len(classes)):
                for inp_x in classes[lab]:
                    ret.x.append(inp_x)
                    ret.y.append(lab)

                    self.data.x.insert(0, inp_x)
                    self.data.y.insert(0, lab)

            FLAGS.total_examples += ret.get_length() - FLAGS.train_examples
            FLAGS.train_examples = ret.get_length()

        ret.shuffle()

        return ret

    def load_testing_data(self):
        ret = Dataset([], [], FLAGS.random_state)

        for i in range(FLAGS.test_examples):
            ret.x.append(self.eval_data.x[i])
            ret.y.append(self.eval_data.y[i])

        return ret

    def post_process_flags(self):
        FLAGS.train_examples = self.data.get_length()
        FLAGS.test_examples = self.eval_data.get_length()
        FLAGS.total_examples = FLAGS.train_examples + FLAGS.test_examples

    @staticmethod
    def load_external_raw():
        train_data, test_data, vocab = None, None, None

        if FLAGS.refresh_data:
            train_data = DataLoader.parse_json(FLAGS.raw_data_loc)
            dj_eval_data = DataLoader.parse_json(FLAGS.raw_dj_eval_loc)

            train_txt = [z[0] for z in train_data]
            eval_txt = [z[0] for z in dj_eval_data]

            train_lab = [z[1] for z in train_data]
            eval_lab = [z[1] for z in dj_eval_data]

            tf.logging.info('Loading preprocessing dependencies')
            load_dependencies()

            tf.logging.info('Processing train data')
            train_txt, train_pos, train_sent = process_dataset(train_txt)
            tf.logging.info('Processing eval data')
            eval_txt, eval_pos, eval_sent = process_dataset(eval_txt)

            if not FLAGS.elmo_embed:
                tokenizer = Tokenizer()

                tokenizer.fit_on_texts(np.concatenate((train_txt, eval_txt)))
                train_seq = tokenizer.texts_to_sequences(train_txt)
                eval_seq = tokenizer.texts_to_sequences(eval_txt)

                train_data = Dataset(list(zip(train_seq, train_pos, train_sent)), train_lab, random_state=FLAGS.random_state)
                eval_data = Dataset(list(zip(eval_seq, eval_pos, eval_sent)), eval_lab, random_state=FLAGS.random_state)
                vocab = tokenizer.word_index
            else:
                train_txt = [z.split(' ') for z in train_txt]
                eval_txt = [z.split(' ') for z in eval_txt]

                train_data = Dataset(list(zip(train_txt, train_pos, train_sent)), train_lab, random_state=FLAGS.random_state)
                eval_data = Dataset(list(zip(eval_txt, eval_pos, eval_sent)), eval_lab, random_state=FLAGS.random_state)

            with open(FLAGS.prc_data_loc, 'wb') as f:
                pickle.dump((train_data, eval_data, vocab), f)
            tf.logging.info('Refreshed data, successfully dumped at {}'.format(FLAGS.prc_data_loc))

            if os.path.exists(FLAGS.output_dir):
                shutil.rmtree(FLAGS.output_dir)
            os.mkdir(FLAGS.output_dir)
        else:
            tf.logging.info('Restoring data from {}'.format(FLAGS.prc_data_loc))
            with open(FLAGS.prc_data_loc, 'rb') as f:
                train_data, eval_data, vocab = pickle.load(f)

        if not FLAGS.elmo_embed:
            assert vocab is not None

        return train_data, eval_data, vocab

    @staticmethod
    def parse_json(json_loc):
        with open(json_loc) as f:
            temp_data = json.load(f)

        dl = []
        labels = [0, 0, 0]

        for el in temp_data:
            lab = int(el["label"]) + 1
            txt = el["text"]

            labels[lab] += 1
            dl.append([txt, lab])

        print('{}: {}'.format(json_loc, labels))
        return dl


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


def get_tags(sentence):
    text = nltk.tokenize.word_tokenize(sentence)
    res = nltk.pos_tag(text)
    return res


def char_list_to_string(lst):
    ret = ""
    for f in lst:
        ret = ret + str(f)
    return ret


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


class ClaimBusterModel:
    def __init__(self, vocab=None, cls_weights=None, restore=False):
        self.x_nl = tf.placeholder(tf.int32, (None, FLAGS.max_len), name='x_nl') if not FLAGS.elmo_embed \
            else tf.placeholder(tf.string, (None, None))
        self.x_pos = tf.placeholder(tf.int32, (None, FLAGS.max_len, len(pos_labels) + 1), name='x_pos')
        self.x_sent = tf.placeholder(tf.float32, (None, 2), name='x_sent')

        self.nl_len = tf.placeholder(tf.int32, (None,), name='nl_len')
        self.pos_len = tf.placeholder(tf.int32, (None,), name='pos_len')

        self.nl_output_mask = tf.placeholder(tf.bool, (None, FLAGS.max_len), name='nl_output_mask')
        self.pos_output_mask = tf.placeholder(tf.bool, (None, FLAGS.max_len), name='pos_output_mask')

        self.y = tf.placeholder(tf.int32, (None, FLAGS.num_classes), name='y')

        self.kp_cls = tf.placeholder(tf.float32, name='kp_cls')
        self.kp_lstm = tf.placeholder(tf.float32, name='kp_lstm')
        self.cls_weight = tf.placeholder(tf.float32, (None,), name='cls_weight')

        self.computed_cls_weights = cls_weights if cls_weights is not None else [1 for _ in range(FLAGS.num_classes)]

        if not restore:
            exit()
        else:
            self.cost, self.y_pred, self.acc = None, None, None

    def get_preds(self, sess, sentence_tuple):
        x_nl = self.pad_seq([sentence_tuple[0]], ver=(0 if not FLAGS.elmo_embed else 1))
        x_pos = self.prc_pos(self.pad_seq([sentence_tuple[1]]))
        x_sent = [sentence_tuple[2]]

        feed_dict = {
            self.x_nl: x_nl,
            self.x_pos: x_pos,
            self.x_sent: x_sent,

            self.nl_len: self.gen_x_len(x_nl),
            self.pos_len: self.gen_x_len(x_pos),

            self.nl_output_mask: self.gen_output_mask(x_nl),
            self.pos_output_mask: self.gen_output_mask(x_pos),

            self.kp_cls: 1.0,
            self.kp_lstm: 1.0,
        }

        return sess.run(self.y_pred, feed_dict=feed_dict)

    def get_cls_weights(self, batch_y):
        return [self.computed_cls_weights[z] for z in batch_y]

    @staticmethod
    def prc_pos(pos_data):
        ret = np.zeros(shape=(len(pos_data), FLAGS.max_len, len(pos_labels) + 1))

        for i in range(len(pos_data)):
            sentence = pos_data[i]
            for j in range(len(sentence)):
                code = sentence[j] + 1
                ret[i][j][code] = 1

        return ret

    @staticmethod
    def pad_seq(inp, ver=0):  # 0 is int, 1 is string
        return pad_sequences(inp, padding="post", maxlen=FLAGS.max_len) if ver == 0 else \
            pad_sequences(inp, padding="post", maxlen=FLAGS.max_len, dtype='str', value='')

    @staticmethod
    def one_hot(a, nc=FLAGS.num_classes):
        return to_categorical(a, num_classes=nc)

    @staticmethod
    def gen_output_mask(inp):
        return [[1 if j == len(el) - 1 else 0 for j in range(FLAGS.max_len)] for el in inp]

    @staticmethod
    def gen_x_len(inp):
        return [len(el) for el in inp]

    @staticmethod
    def save_model(sess, epoch):
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(FLAGS.output_dir, 'cb.ckpt'), global_step=epoch)

    @staticmethod
    def transform_dl_data(data_xlist):
        temp = [[z[0] for z in data_xlist], [z[1] for z in data_xlist]]
        return np.swapaxes(temp, 0, 1)

    @staticmethod
    def get_batch(bid, data, ver='train'):
        batch_x = []
        batch_y = []

        for i in range(FLAGS.batch_size):
            idx = bid * FLAGS.batch_size + i
            if idx >= (FLAGS.train_examples if ver == 'train' else FLAGS.test_examples):
                break

            batch_x.append(list(data.x[idx]))
            batch_y.append(data.y[idx])

        return batch_x, batch_y

    def load_model(self, sess, graph):
        def get_last_save(scan_loc):
            ret_ar = []
            directory = os.fsencode(scan_loc)
            for fstr in os.listdir(directory):
                if '.meta' in os.fsdecode(fstr) and 'cb.ckpt-' in os.fsdecode(fstr):
                    ret_ar.append(os.fsdecode(fstr))
            ret_ar.sort()
            return ret_ar[-1]

        model_dir = os.path.join(FLAGS.output_dir, get_last_save(FLAGS.output_dir))
        tf.logging.info('Attempting to restore from {}'.format(model_dir))

        with graph.as_default():
            saver = tf.train.import_meta_graph(model_dir)
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.output_dir))

            # inputs
            self.x_nl = graph.get_tensor_by_name('x_nl:0')
            self.x_pos = graph.get_tensor_by_name('x_pos:0')
            self.x_sent = graph.get_tensor_by_name('x_sent:0')

            self.nl_len = graph.get_tensor_by_name('nl_len:0')
            self.pos_len = graph.get_tensor_by_name('pos_len:0')

            self.nl_output_mask = graph.get_tensor_by_name('nl_output_mask:0')
            self.pos_output_mask = graph.get_tensor_by_name('pos_output_mask:0')

            self.y = graph.get_tensor_by_name('y:0')

            self.kp_cls = graph.get_tensor_by_name('kp_cls:0')
            self.kp_lstm = graph.get_tensor_by_name('kp_lstm:0')
            self.cls_weight = graph.get_tensor_by_name('cls_weight:0')

            # outputs
            self.cost = graph.get_tensor_by_name('cb_model/cost:0')
            self.y_pred = graph.get_tensor_by_name('y_pred:0')
            self.acc = graph.get_tensor_by_name('acc:0')

            tf.logging.info('Model successfully restored.')


return_strings = ['Non-factual statement', 'Unimportant factual statement', 'Salient factual statement']


def prc_sentence(sentence, vocab):
    def get_idx(str):
        if str in vocab:
            return vocab[str]
        else:
            return 0

    sentence = transform_sentence_complete(sentence)

    sent = get_sentiment(sentence)
    pos = process_sentence_full_tags(sentence)
    sentence = [get_idx(z) for z in sentence.split(' ')]

    return sentence, pos, sent


def subscribe_query(sess, cb_model, vocab):
    print('Enter a sentence to process')
    sentence_tuple = prc_sentence(input().strip('\n\r\t '), vocab)
    return cb_model.get_preds(sess, sentence_tuple)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(z) for z in FLAGS.gpu_active])

    data_load = DataLoader()
    vocab = data_load.vocab

    load_dependencies()
    cb_model = ClaimBusterModel(restore=True)

    graph = tf.Graph()
    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        cb_model.load_model(sess, graph)

        while True:
            res = subscribe_query(sess, cb_model, vocab)
            print('Probability of CFS: {}'.format(res[0][2]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
