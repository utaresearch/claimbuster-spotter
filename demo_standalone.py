import tensorflow as tf
import numpy as np
import os
import time
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from utils.data_loader import DataLoader
from models.recurrent import RecurrentModel
from models.embeddings import Embedding
from sklearn.metrics import f1_score
import math
from flags import FLAGS
import nltk
import spacy
import string
from textblob import TextBlob
from keras.preprocessing.text import text_to_word_sequence
from tqdm import tqdm

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
            if not FLAGS.elmo_embed:
                self.embed_obj = Embedding(vocab)
                self.embed = self.embed_obj.construct_embeddings()

            self.logits, self.cost = self.construct_model(adv=FLAGS.adv_train)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.cost) \
                if FLAGS.adam else tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.cost)

            self.y_pred = tf.nn.softmax(self.logits, axis=1, name='y_pred')
            self.correct = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(self.y_pred, axis=1))
            self.acc = tf.reduce_mean(tf.cast(self.correct, tf.float32), name='acc')
        else:
            self.cost, self.y_pred, self.acc = None, None, None

    def construct_model(self, adv):
        with tf.variable_scope('cb_model/'):
            orig_embed, logits = self.fprop()
            loss = self.ce_loss(logits, self.cls_weight)

            if adv:
                logits_adv = self.fprop(orig_embed, loss, adv=True)
                loss += FLAGS.adv_coeff * self.adv_loss(logits_adv, self.cls_weight)

            return logits, tf.identity(loss, name='cost')

    def fprop(self, orig_embed=None, reg_loss=None, adv=False):
        if adv: assert (reg_loss is not None and orig_embed is not None)

        with tf.variable_scope('natural_lang_lstm/', reuse=adv):
            nl_lstm_out = RecurrentModel.build_embed_lstm(self.x_nl, self.nl_len, self.nl_output_mask, self.embed,
                                                          self.kp_lstm, orig_embed, reg_loss, adv) \
                if not FLAGS.elmo_embed else RecurrentModel.build_bert_lstm(self.x_nl, self.nl_len,
                                                                            self.nl_output_mask,
                                                                            self.kp_lstm, orig_embed, reg_loss, adv)
            if not adv:
                orig_embed, nl_lstm_out = nl_lstm_out

        with tf.variable_scope('pos_lstm/', reuse=adv):
            pos_lstm_out = RecurrentModel.build_lstm(self.x_pos, self.pos_len, self.pos_output_mask, self.kp_lstm,
                                                     adv)

        with tf.variable_scope('fc_output/', reuse=adv):
            lstm_out = tf.concat([nl_lstm_out, pos_lstm_out, self.x_sent], axis=1)
            lstm_out = tf.nn.dropout(lstm_out, keep_prob=FLAGS.keep_prob_cls)

            output_weights = tf.get_variable('cb_output_weights', shape=(lstm_out.get_shape()[1], FLAGS.num_classes),
                                             initializer=tf.contrib.layers.xavier_initializer())
            output_biases = tf.get_variable('cb_output_biases', shape=FLAGS.num_classes,
                                            initializer=tf.zeros_initializer())

            cb_out = tf.matmul(lstm_out, output_weights) + output_biases

        return (orig_embed, cb_out) if not adv else cb_out

    def adv_loss(self, logits, cls_weight):
        return tf.identity(self.ce_loss(logits, cls_weight), name='adv_loss')

    def ce_loss(self, logits, cls_weight):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits)
        loss_l2 = 0

        # if FLAGS.l2_reg_coeff > 0.0:
        #     varlist = tf.trainable_variables()
        #     loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in varlist if 'bias' not in v.name]) * FLAGS.l2_reg_coeff

        ret_loss = loss + loss_l2

        if FLAGS.weight_classes_loss:
            ret_loss *= cls_weight

        return tf.identity(ret_loss, name='regular_loss')

    def train_neural_network(self, sess, batch_x, batch_y):
        x_nl = [z[0] for z in batch_x]
        x_pos = [z[1] for z in batch_x]
        x_sent = [z[2] for z in batch_x]

        sess.run(
            self.optimizer,
            feed_dict={
                self.x_nl: self.pad_seq(x_nl, ver=(0 if not FLAGS.elmo_embed else 1)),
                self.x_pos: self.prc_pos(self.pad_seq(x_pos)),
                self.x_sent: x_sent,

                self.nl_len: self.gen_x_len(x_nl),
                self.pos_len: self.gen_x_len(x_pos),

                self.nl_output_mask: self.gen_output_mask(x_nl),
                self.pos_output_mask: self.gen_output_mask(x_pos),

                self.y: self.one_hot(batch_y),

                self.kp_cls: 1.0,
                self.kp_lstm: 1.0,
                self.cls_weight: self.get_cls_weights(batch_y)
            }
        )

    def execute_validation(self, sess, test_data):
        n_batches = math.ceil(float(FLAGS.test_examples) / float(FLAGS.batch_size))
        val_loss, val_acc = 0.0, 0.0
        tot_val_ex = 0

        all_y_pred = []
        all_y = []
        for batch in range(n_batches):
            batch_x, batch_y = self.get_batch(batch, test_data, ver='validation')
            tloss, tacc, tpred = self.stats_from_run(sess, batch_x, batch_y)

            val_loss += tloss
            val_acc += tacc * len(batch_y)
            tot_val_ex += len(batch_y)

            all_y_pred = np.concatenate((all_y_pred, tpred))
            all_y = np.concatenate((all_y, batch_y))

        val_loss /= tot_val_ex
        val_acc /= tot_val_ex
        val_f1 = f1_score(all_y, all_y_pred, average='weighted')

        return 'DJ Val Loss: {:>7.4f} DJ Val F1: {:>7.4f} '.format(val_loss, val_f1)

    def stats_from_run(self, sess, batch_x, batch_y):
        x_nl = [z[0] for z in batch_x]
        x_pos = [z[1] for z in batch_x]
        x_sent = [z[2] for z in batch_x]

        feed_dict = {
            self.x_nl: self.pad_seq(x_nl, ver=(0 if not FLAGS.elmo_embed else 1)),
            self.x_pos: self.prc_pos(self.pad_seq(x_pos)),
            self.x_sent: x_sent,

            self.nl_len: self.gen_x_len(x_nl),
            self.pos_len: self.gen_x_len(x_pos),

            self.nl_output_mask: self.gen_output_mask(x_nl),
            self.pos_output_mask: self.gen_output_mask(x_pos),

            self.y: self.one_hot(batch_y),

            self.kp_cls: 1.0,
            self.kp_lstm: 1.0,
            self.cls_weight: self.get_cls_weights(batch_y)
        }

        run_loss = sess.run(self.cost, feed_dict=feed_dict)
        run_acc = sess.run(self.acc, feed_dict=feed_dict)
        run_pred = sess.run(self.y_pred, feed_dict=feed_dict)

        return np.sum(run_loss), run_acc, np.argmax(run_pred, axis=1)

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

    sent = get_sentiment(sentence)
    pos = process_sentence_full_tags(sentence)
    sentence = [get_idx(z) for z in transform_sentence_complete(sentence).split(' ')]

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
            idx = np.argmax(res, axis=1)

            print('Probability of CFS: {}'.format(res[2]))

            # print('{} with probability {}'.format(np.array(return_strings)[idx][0], res[0][idx][0]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
