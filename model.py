import tensorflow as tf
from bert import optimization
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from models.lang_model import LanguageModel
from models.embeddings import Embedding
from sklearn.metrics import f1_score
from utils.transformations import pos_labels
import math
from flags import FLAGS


class ClaimBusterModel:
    def __init__(self, vocab=None, cls_weights=None, restore=False):
        self.x_nl = tf.placeholder(tf.int32, (None, FLAGS.max_len), name='x_nl') if not FLAGS.bert_model \
            else [tf.placeholder(tf.int32, (None, FLAGS.max_len), name='x_id'),
                  tf.placeholder(tf.int32, (None, FLAGS.max_len), name='x_mask'),
                  tf.placeholder(tf.int32, (None, FLAGS.max_len), name='x_segment')]
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
            if not FLAGS.bert_model:
                self.embed_obj = Embedding(vocab)
                self.embed = self.embed_obj.construct_embeddings()

            self.logits, self.cost = self.construct_model(adv=FLAGS.adv_train)

            self.optimizer = self.build_optimizer()

            self.y_pred = tf.nn.softmax(self.logits, axis=1, name='y_pred')
            self.correct = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(self.y_pred, axis=1))
            self.acc = tf.reduce_mean(tf.cast(self.correct, tf.float32), name='acc')
        else:
            self.cost, self.y_pred, self.acc = None, None, None

    def build_optimizer(self):
        if not FLAGS.bert_model:
            return tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.cost) \
                if FLAGS.adam else tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.cost)
        else:
            train_vars = tf.trainable_variables()
            non_trainable_layers = ['layer_{}'.format(num)
                                    for num in range(FLAGS.bert_layers - FLAGS.bert_fine_tune_layers)]

            tf.logging.info(non_trainable_layers)
            tf.logging.info(train_vars)

            orig_train_vars = train_vars.copy()

            if FLAGS.bert_trainable:
                train_vars = [v for v in train_vars
                              if not any(z in v.name for z in non_trainable_layers)]

            tf.logging.info(train_vars)

            tf.logging.info(' ')

            tf.logging.info(list(set(orig_train_vars).difference(set(train_vars))))

            return tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.cost, var_list=train_vars)\
                if FLAGS.adam else tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.cost)

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
            nl_lstm_out = LanguageModel.build_embed_lstm(self.x_nl, self.nl_len, self.nl_output_mask, self.embed,
                                                         self.kp_lstm, orig_embed, reg_loss, adv) \
                if not FLAGS.bert_model else LanguageModel.build_bert_transformer(self.x_nl[0], self.x_nl[1],
                                                                                  self.x_nl[2], adv)
            if not adv:
                orig_embed, nl_lstm_out = nl_lstm_out

        with tf.variable_scope('pos_lstm/', reuse=adv):
            if FLAGS.pos_lstm:
                pos_lstm_out = LanguageModel.build_lstm(self.x_pos, self.pos_len, self.pos_output_mask, self.kp_lstm,
                                                        adv)

        with tf.variable_scope('fc_output/', reuse=adv):
            to_concat = [nl_lstm_out, pos_lstm_out, self.x_sent] if FLAGS.pos_lstm else [nl_lstm_out, self.x_sent]

            lstm_out = tf.concat(to_concat, axis=1)
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

        if FLAGS.l2_reg_coeff > 0.0:
            varlist = tf.trainable_variables()
            loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in varlist if 'bias' not in v.name]) * FLAGS.l2_reg_coeff

        ret_loss = loss + loss_l2

        if FLAGS.weight_classes_loss:
            ret_loss *= cls_weight

        return tf.identity(ret_loss, name='regular_loss')

    def get_feed_dict(self, x_nl, x_pos, x_sent, batch_y=None, ver='train'):
        if not FLAGS.bert_model:
            feed_dict = {
                self.x_nl: self.pad_seq(x_nl),
                self.x_pos: self.prc_pos(self.pad_seq(x_pos)),
                self.x_sent: x_sent,

                self.nl_len: self.gen_x_len(x_nl),
                self.pos_len: self.gen_x_len(x_pos),

                self.nl_output_mask: self.gen_output_mask(x_nl),
                self.pos_output_mask: self.gen_output_mask(x_pos),

                self.kp_cls: FLAGS.keep_prob_cls if ver == 'train' else 1.0,
                self.kp_lstm: FLAGS.keep_prob_lstm if ver == 'train' else 1.0,
            }
        else:
            feed_dict = {
                self.x_nl[0]: [z.input_ids for z in x_nl],
                self.x_nl[1]: [z.input_mask for z in x_nl],
                self.x_nl[2]: [z.segment_ids for z in x_nl],
                self.x_pos: self.prc_pos(self.pad_seq(x_pos)),
                self.x_sent: x_sent,

                self.pos_len: self.gen_x_len(x_pos),
                self.pos_output_mask: self.gen_output_mask(x_pos),

                self.kp_cls: FLAGS.keep_prob_cls if ver == 'train' else 1.0,
                self.kp_lstm: FLAGS.keep_prob_lstm if ver == 'train' else 1.0,
            }

        if batch_y is not None:
            feed_dict[self.y] = self.one_hot(batch_y)
            feed_dict[self.cls_weight] = self.get_cls_weights(batch_y)

        return feed_dict

    def train_neural_network(self, sess, batch_x, batch_y):
        x_nl = [z[0] for z in batch_x]
        x_pos = [z[1] for z in batch_x]
        x_sent = [z[2] for z in batch_x]

        feed_dict = self.get_feed_dict(x_nl, x_pos, x_sent, batch_y, ver='train')

        sess.run(
            self.optimizer,
            feed_dict=feed_dict
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

        feed_dict = self.get_feed_dict(x_nl, x_pos, x_sent, batch_y, ver='test')

        run_loss = sess.run(self.cost, feed_dict=feed_dict)
        run_acc = sess.run(self.acc, feed_dict=feed_dict)
        run_pred = sess.run(self.y_pred, feed_dict=feed_dict)

        return np.sum(run_loss), run_acc, np.argmax(run_pred, axis=1)

    def get_preds(self, sess, sentence_tuple):
        x_nl = self.pad_seq([sentence_tuple[0]], ver=(0 if not FLAGS.bert_model else 1))
        x_pos = self.prc_pos(self.pad_seq([sentence_tuple[1]]))
        x_sent = [sentence_tuple[2]]

        feed_dict = self.get_feed_dict(x_nl, x_pos, x_sent, ver='test')

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
        saver.save(sess, os.path.join(FLAGS.cb_output_dir, 'cb.ckpt'), global_step=epoch)

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

        model_dir = os.path.join(FLAGS.cb_output_dir, get_last_save(FLAGS.cb_output_dir))
        tf.logging.info('Attempting to restore from {}'.format(model_dir))

        with graph.as_default():
            saver = tf.train.import_meta_graph(model_dir)

            print(FLAGS.cb_output_dir)

            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.cb_output_dir))

            # inputs
            if not FLAGS.bert_model:
                self.x_nl = graph.get_tensor_by_name('x_nl:0')
                self.nl_len = graph.get_tensor_by_name('nl_len:0')
                self.nl_output_mask = graph.get_tensor_by_name('nl_output_mask:0')
            else:
                self.x_nl = [graph.get_tensor_by_name('x_id:0'), graph.get_tensor_by_name('x_mask:0'),
                             graph.get_tensor_by_name('x_segment:0')]

            self.x_pos = graph.get_tensor_by_name('x_pos:0')
            self.x_sent = graph.get_tensor_by_name('x_sent:0')

            self.pos_len = graph.get_tensor_by_name('pos_len:0')
            self.pos_output_mask = graph.get_tensor_by_name('pos_output_mask:0')

            self.y = graph.get_tensor_by_name('y:0')

            self.kp_cls = graph.get_tensor_by_name('kp_cls:0')
            self.kp_lstm = graph.get_tensor_by_name('kp_lstm:0')
            self.cls_weight = graph.get_tensor_by_name('cls_weight:0')

            # outputs
            self.cost = graph.get_tensor_by_name('cb_model/cost:0')
            self.y_pred = graph.get_tensor_by_name('y_pred:0')
            self.acc = graph.get_tensor_by_name('acc:0')

            for v in tf.trainable_variables():
                print(v.name)

            tf.logging.info('Model successfully restored.')