import tensorflow as tf
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import f1_score
import math

cwd = os.getcwd()
root_dir = None

for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.endswith("ac_bert.txt"):
            root_dir = root

if cwd != root_dir:
    from .models.lang_model import LanguageModel
    from .utils.transformations import pos_labels
    from .flags import FLAGS
else:
    from models.lang_model import LanguageModel
    from utils.transformations import pos_labels
    from flags import FLAGS


class ClaimBusterModel:
    def __init__(self, vocab=None, cls_weights=None, restore=False):
        self.x_nl = [tf.placeholder(tf.int32, (None, FLAGS.max_len), name='x_id'),
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

        # self.init_bert_pretrain_op = self.restore_bert_pretrain_weights()

        if not restore:
            self.logits, self.cost = self.construct_model(adv=FLAGS.adv_train)

            self.optimizer = self.build_optimizer()

            self.y_pred = tf.nn.softmax(self.logits, axis=1, name='y_pred')
            self.correct = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(self.y_pred, axis=1))
            self.acc = tf.reduce_mean(tf.cast(self.correct, tf.float32), name='acc')
        else:
            self.cost, self.y_pred, self.acc = None, None, None

    def build_optimizer(self):
        train_vars = tf.trainable_variables()
        non_trainable_layers = ['/layer_{}/'.format(num)
                                for num in range(FLAGS.bert_layers - FLAGS.bert_ft_enc_layers)]
        if not FLAGS.bert_ft_embed:
            non_trainable_layers.append('/embeddings/')

        if FLAGS.bert_trainable:
            train_vars = [v for v in train_vars if not any(z in v.name for z in non_trainable_layers)]

        tf.logging.info('Removing: {}'.format(non_trainable_layers))
        tf.logging.info(train_vars)

        return tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.cost, var_list=train_vars) \
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

        with tf.variable_scope('natural_lang_model/', reuse=adv):
            nl_out = LanguageModel.build_bert_transformer_raw(self.x_nl[0], self.x_nl[1], self.x_nl[2], adv)
            if not adv:
                orig_embed, nl_out = nl_out

        with tf.variable_scope('fc_output/', reuse=adv):
            synth_out = tf.concat([nl_out, self.x_sent], axis=1)
            synth_out = tf.nn.dropout(synth_out, keep_prob=FLAGS.keep_prob_cls)

            synth_out_shape = synth_out.get_shape()[1]

            if FLAGS.cls_hidden > 0:
                hidden_weights = tf.get_variable('cb_hidden_weights', shape=(synth_out_shape, FLAGS.cls_hidden),
                                                 initializer=tf.contrib.layers.xavier_initializer())
                hidden_biases = tf.get_variable('cb_hidden_biases', shape=FLAGS.cls_hidden,
                                                initializer=tf.contrib.layers.xavier_initializer())

                synth_out = tf.matmul(synth_out, hidden_weights) + hidden_biases
                synth_out = tf.nn.dropout(synth_out, keep_prob=FLAGS.keep_prob_cls)

            output_weights = tf.get_variable('cb_output_weights', shape=(
                FLAGS.cls_hidden if FLAGS.cls_hidden > 0 else synth_out_shape, FLAGS.num_classes),
                                             initializer=tf.contrib.layers.xavier_initializer())
            output_biases = tf.get_variable('cb_output_biases', shape=FLAGS.num_classes,
                                            initializer=tf.zeros_initializer())

            cb_out = tf.matmul(synth_out, output_weights) + output_biases

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

    @staticmethod
    def restore_bert_pretrain_weights():
        graph = tf.Graph()

        tensors_to_restore = {'bert/embeddings/word_embeddings:0', 'bert/embeddings/token_type_embeddings:0', 'bert/embeddings/position_embeddings:0', 'bert/embeddings/LayerNorm/beta:0', 'bert/embeddings/LayerNorm/gamma:0', 'bert/encoder/layer_0/attention/self/query/kernel:0', 'bert/encoder/layer_0/attention/self/query/bias:0', 'bert/encoder/layer_0/attention/self/key/kernel:0', 'bert/encoder/layer_0/attention/self/key/bias:0', 'bert/encoder/layer_0/attention/self/value/kernel:0', 'bert/encoder/layer_0/attention/self/value/bias:0', 'bert/encoder/layer_0/attention/output/dense/kernel:0', 'bert/encoder/layer_0/attention/output/dense/bias:0', 'bert/encoder/layer_0/attention/output/LayerNorm/beta:0', 'bert/encoder/layer_0/attention/output/LayerNorm/gamma:0', 'bert/encoder/layer_0/intermediate/dense/kernel:0', 'bert/encoder/layer_0/intermediate/dense/bias:0', 'bert/encoder/layer_0/output/dense/kernel:0', 'bert/encoder/layer_0/output/dense/bias:0', 'bert/encoder/layer_0/output/LayerNorm/beta:0', 'bert/encoder/layer_0/output/LayerNorm/gamma:0', 'bert/encoder/layer_1/attention/self/query/kernel:0', 'bert/encoder/layer_1/attention/self/query/bias:0', 'bert/encoder/layer_1/attention/self/key/kernel:0', 'bert/encoder/layer_1/attention/self/key/bias:0', 'bert/encoder/layer_1/attention/self/value/kernel:0', 'bert/encoder/layer_1/attention/self/value/bias:0', 'bert/encoder/layer_1/attention/output/dense/kernel:0', 'bert/encoder/layer_1/attention/output/dense/bias:0', 'bert/encoder/layer_1/attention/output/LayerNorm/beta:0', 'bert/encoder/layer_1/attention/output/LayerNorm/gamma:0', 'bert/encoder/layer_1/intermediate/dense/kernel:0', 'bert/encoder/layer_1/intermediate/dense/bias:0', 'bert/encoder/layer_1/output/dense/kernel:0', 'bert/encoder/layer_1/output/dense/bias:0', 'bert/encoder/layer_1/output/LayerNorm/beta:0', 'bert/encoder/layer_1/output/LayerNorm/gamma:0', 'bert/encoder/layer_2/attention/self/query/kernel:0', 'bert/encoder/layer_2/attention/self/query/bias:0', 'bert/encoder/layer_2/attention/self/key/kernel:0', 'bert/encoder/layer_2/attention/self/key/bias:0', 'bert/encoder/layer_2/attention/self/value/kernel:0', 'bert/encoder/layer_2/attention/self/value/bias:0', 'bert/encoder/layer_2/attention/output/dense/kernel:0', 'bert/encoder/layer_2/attention/output/dense/bias:0', 'bert/encoder/layer_2/attention/output/LayerNorm/beta:0', 'bert/encoder/layer_2/attention/output/LayerNorm/gamma:0', 'bert/encoder/layer_2/intermediate/dense/kernel:0', 'bert/encoder/layer_2/intermediate/dense/bias:0', 'bert/encoder/layer_2/output/dense/kernel:0', 'bert/encoder/layer_2/output/dense/bias:0', 'bert/encoder/layer_2/output/LayerNorm/beta:0', 'bert/encoder/layer_2/output/LayerNorm/gamma:0', 'bert/encoder/layer_3/attention/self/query/kernel:0', 'bert/encoder/layer_3/attention/self/query/bias:0', 'bert/encoder/layer_3/attention/self/key/kernel:0', 'bert/encoder/layer_3/attention/self/key/bias:0', 'bert/encoder/layer_3/attention/self/value/kernel:0', 'bert/encoder/layer_3/attention/self/value/bias:0', 'bert/encoder/layer_3/attention/output/dense/kernel:0', 'bert/encoder/layer_3/attention/output/dense/bias:0', 'bert/encoder/layer_3/attention/output/LayerNorm/beta:0', 'bert/encoder/layer_3/attention/output/LayerNorm/gamma:0', 'bert/encoder/layer_3/intermediate/dense/kernel:0', 'bert/encoder/layer_3/intermediate/dense/bias:0', 'bert/encoder/layer_3/output/dense/kernel:0', 'bert/encoder/layer_3/output/dense/bias:0', 'bert/encoder/layer_3/output/LayerNorm/beta:0', 'bert/encoder/layer_3/output/LayerNorm/gamma:0', 'bert/encoder/layer_4/attention/self/query/kernel:0', 'bert/encoder/layer_4/attention/self/query/bias:0', 'bert/encoder/layer_4/attention/self/key/kernel:0', 'bert/encoder/layer_4/attention/self/key/bias:0', 'bert/encoder/layer_4/attention/self/value/kernel:0', 'bert/encoder/layer_4/attention/self/value/bias:0', 'bert/encoder/layer_4/attention/output/dense/kernel:0', 'bert/encoder/layer_4/attention/output/dense/bias:0', 'bert/encoder/layer_4/attention/output/LayerNorm/beta:0', 'bert/encoder/layer_4/attention/output/LayerNorm/gamma:0', 'bert/encoder/layer_4/intermediate/dense/kernel:0', 'bert/encoder/layer_4/intermediate/dense/bias:0', 'bert/encoder/layer_4/output/dense/kernel:0', 'bert/encoder/layer_4/output/dense/bias:0', 'bert/encoder/layer_4/output/LayerNorm/beta:0', 'bert/encoder/layer_4/output/LayerNorm/gamma:0', 'bert/encoder/layer_5/attention/self/query/kernel:0', 'bert/encoder/layer_5/attention/self/query/bias:0', 'bert/encoder/layer_5/attention/self/key/kernel:0', 'bert/encoder/layer_5/attention/self/key/bias:0', 'bert/encoder/layer_5/attention/self/value/kernel:0', 'bert/encoder/layer_5/attention/self/value/bias:0', 'bert/encoder/layer_5/attention/output/dense/kernel:0', 'bert/encoder/layer_5/attention/output/dense/bias:0', 'bert/encoder/layer_5/attention/output/LayerNorm/beta:0', 'bert/encoder/layer_5/attention/output/LayerNorm/gamma:0', 'bert/encoder/layer_5/intermediate/dense/kernel:0', 'bert/encoder/layer_5/intermediate/dense/bias:0', 'bert/encoder/layer_5/output/dense/kernel:0', 'bert/encoder/layer_5/output/dense/bias:0', 'bert/encoder/layer_5/output/LayerNorm/beta:0', 'bert/encoder/layer_5/output/LayerNorm/gamma:0', 'bert/encoder/layer_6/attention/self/query/kernel:0', 'bert/encoder/layer_6/attention/self/query/bias:0', 'bert/encoder/layer_6/attention/self/key/kernel:0', 'bert/encoder/layer_6/attention/self/key/bias:0', 'bert/encoder/layer_6/attention/self/value/kernel:0', 'bert/encoder/layer_6/attention/self/value/bias:0', 'bert/encoder/layer_6/attention/output/dense/kernel:0', 'bert/encoder/layer_6/attention/output/dense/bias:0', 'bert/encoder/layer_6/attention/output/LayerNorm/beta:0', 'bert/encoder/layer_6/attention/output/LayerNorm/gamma:0', 'bert/encoder/layer_6/intermediate/dense/kernel:0', 'bert/encoder/layer_6/intermediate/dense/bias:0', 'bert/encoder/layer_6/output/dense/kernel:0', 'bert/encoder/layer_6/output/dense/bias:0', 'bert/encoder/layer_6/output/LayerNorm/beta:0', 'bert/encoder/layer_6/output/LayerNorm/gamma:0', 'bert/encoder/layer_7/attention/self/query/kernel:0', 'bert/encoder/layer_7/attention/self/query/bias:0', 'bert/encoder/layer_7/attention/self/key/kernel:0', 'bert/encoder/layer_7/attention/self/key/bias:0', 'bert/encoder/layer_7/attention/self/value/kernel:0', 'bert/encoder/layer_7/attention/self/value/bias:0', 'bert/encoder/layer_7/attention/output/dense/kernel:0', 'bert/encoder/layer_7/attention/output/dense/bias:0', 'bert/encoder/layer_7/attention/output/LayerNorm/beta:0', 'bert/encoder/layer_7/attention/output/LayerNorm/gamma:0', 'bert/encoder/layer_7/intermediate/dense/kernel:0', 'bert/encoder/layer_7/intermediate/dense/bias:0', 'bert/encoder/layer_7/output/dense/kernel:0', 'bert/encoder/layer_7/output/dense/bias:0', 'bert/encoder/layer_7/output/LayerNorm/beta:0', 'bert/encoder/layer_7/output/LayerNorm/gamma:0', 'bert/encoder/layer_8/attention/self/query/kernel:0', 'bert/encoder/layer_8/attention/self/query/bias:0', 'bert/encoder/layer_8/attention/self/key/kernel:0', 'bert/encoder/layer_8/attention/self/key/bias:0', 'bert/encoder/layer_8/attention/self/value/kernel:0', 'bert/encoder/layer_8/attention/self/value/bias:0', 'bert/encoder/layer_8/attention/output/dense/kernel:0', 'bert/encoder/layer_8/attention/output/dense/bias:0', 'bert/encoder/layer_8/attention/output/LayerNorm/beta:0', 'bert/encoder/layer_8/attention/output/LayerNorm/gamma:0', 'bert/encoder/layer_8/intermediate/dense/kernel:0', 'bert/encoder/layer_8/intermediate/dense/bias:0', 'bert/encoder/layer_8/output/dense/kernel:0', 'bert/encoder/layer_8/output/dense/bias:0', 'bert/encoder/layer_8/output/LayerNorm/beta:0', 'bert/encoder/layer_8/output/LayerNorm/gamma:0', 'bert/encoder/layer_9/attention/self/query/kernel:0', 'bert/encoder/layer_9/attention/self/query/bias:0', 'bert/encoder/layer_9/attention/self/key/kernel:0', 'bert/encoder/layer_9/attention/self/key/bias:0', 'bert/encoder/layer_9/attention/self/value/kernel:0', 'bert/encoder/layer_9/attention/self/value/bias:0', 'bert/encoder/layer_9/attention/output/dense/kernel:0', 'bert/encoder/layer_9/attention/output/dense/bias:0', 'bert/encoder/layer_9/attention/output/LayerNorm/beta:0', 'bert/encoder/layer_9/attention/output/LayerNorm/gamma:0', 'bert/encoder/layer_9/intermediate/dense/kernel:0', 'bert/encoder/layer_9/intermediate/dense/bias:0', 'bert/encoder/layer_9/output/dense/kernel:0', 'bert/encoder/layer_9/output/dense/bias:0', 'bert/encoder/layer_9/output/LayerNorm/beta:0', 'bert/encoder/layer_9/output/LayerNorm/gamma:0', 'bert/encoder/layer_10/attention/self/query/kernel:0', 'bert/encoder/layer_10/attention/self/query/bias:0', 'bert/encoder/layer_10/attention/self/key/kernel:0', 'bert/encoder/layer_10/attention/self/key/bias:0', 'bert/encoder/layer_10/attention/self/value/kernel:0', 'bert/encoder/layer_10/attention/self/value/bias:0', 'bert/encoder/layer_10/attention/output/dense/kernel:0', 'bert/encoder/layer_10/attention/output/dense/bias:0', 'bert/encoder/layer_10/attention/output/LayerNorm/beta:0', 'bert/encoder/layer_10/attention/output/LayerNorm/gamma:0', 'bert/encoder/layer_10/intermediate/dense/kernel:0', 'bert/encoder/layer_10/intermediate/dense/bias:0', 'bert/encoder/layer_10/output/dense/kernel:0', 'bert/encoder/layer_10/output/dense/bias:0', 'bert/encoder/layer_10/output/LayerNorm/beta:0', 'bert/encoder/layer_10/output/LayerNorm/gamma:0', 'bert/encoder/layer_11/attention/self/query/kernel:0', 'bert/encoder/layer_11/attention/self/query/bias:0', 'bert/encoder/layer_11/attention/self/key/kernel:0', 'bert/encoder/layer_11/attention/self/key/bias:0', 'bert/encoder/layer_11/attention/self/value/kernel:0', 'bert/encoder/layer_11/attention/self/value/bias:0', 'bert/encoder/layer_11/attention/output/dense/kernel:0', 'bert/encoder/layer_11/attention/output/dense/bias:0', 'bert/encoder/layer_11/attention/output/LayerNorm/beta:0', 'bert/encoder/layer_11/attention/output/LayerNorm/gamma:0', 'bert/encoder/layer_11/intermediate/dense/kernel:0', 'bert/encoder/layer_11/intermediate/dense/bias:0', 'bert/encoder/layer_11/output/dense/kernel:0', 'bert/encoder/layer_11/output/dense/bias:0', 'bert/encoder/layer_11/output/LayerNorm/beta:0', 'bert/encoder/layer_11/output/LayerNorm/gamma:0', 'bert/pooler/dense/kernel:0', 'bert/pooler/dense/bias:0'}
        ret = dict()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            model_dir = os.path.join(FLAGS.bert_model_loc, 'bert_model.ckpt.meta')
            tf.logging.info('Attempting to restore pretrained BERT weights from from {}'.format(model_dir))

            with graph.as_default():
                saver = tf.train.import_meta_graph(model_dir)
                saver.restore(sess, os.path.join(FLAGS.bert_model_loc, 'bert_model.ckpt'))

                for tensor_name in tensors_to_restore:
                    ret[tensor_name] = sess.run(graph.get_tensor_by_name(tensor_name))

                tf.logging.info('Model weights successfully loaded.')

        tf.reset_default_graph()

        return ret

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

        return 'Dev Loss: {:>7.4f} Dev F1: {:>7.4f} '.format(val_loss, val_f1)

    def stats_from_run(self, sess, batch_x, batch_y):
        x_nl = [z[0] for z in batch_x]
        x_pos = [z[1] for z in batch_x]
        x_sent = [z[2] for z in batch_x]

        feed_dict = self.get_feed_dict(x_nl, x_pos, x_sent, batch_y, ver='test')

        run_loss = sess.run(self.cost, feed_dict=feed_dict)
        run_acc = sess.run(self.acc, feed_dict=feed_dict)
        run_pred = sess.run(self.y_pred, feed_dict=feed_dict)

        return np.sum(run_loss), run_acc, np.argmax(run_pred, axis=1)

    def get_preds(self, sess, inp):
        x_nl, x_pos, x_sent = inp[0], [inp[1]], [inp[2]]
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
            print(scan_loc)
            ret_ar = []
            directory = os.fsencode(scan_loc)
            for fstr in os.listdir(directory):
                print(os.fsdecode(fstr))
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

            tf.logging.info('Model successfully restored.')