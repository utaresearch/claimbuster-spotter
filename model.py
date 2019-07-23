import numpy as np
import collections
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

import tensorflow as tf


class ClaimBusterModel:
    def __init__(self, vocab=None, cls_weights=None, restore=False, adv=False):
        self.x_nl = [tf.placeholder(tf.int32, (None, FLAGS.max_len), name='x_id'),
                     tf.placeholder(tf.int32, (None, FLAGS.max_len), name='x_mask'),
                     tf.placeholder(tf.int32, (None, FLAGS.max_len), name='x_segment')]

        self.adv = adv

        self.x_pos = tf.placeholder(tf.int32, (None, FLAGS.max_len, len(pos_labels) + 1), name='x_pos')
        self.x_sent = tf.placeholder(tf.float32, (None, 2), name='x_sent')

        self.nl_len = tf.placeholder(tf.int32, (None,), name='nl_len')
        self.pos_len = tf.placeholder(tf.int32, (None,), name='pos_len')

        self.nl_output_mask = tf.placeholder(tf.bool, (None, FLAGS.max_len), name='nl_output_mask')
        self.pos_output_mask = tf.placeholder(tf.bool, (None, FLAGS.max_len), name='pos_output_mask')

        self.y = tf.placeholder(tf.int32, (None, FLAGS.num_classes), name='y')

        self.kp_cls = tf.placeholder(tf.float32, name='kp_cls')
        self.kp_tfm_atten = tf.placeholder(tf.float32, name='kp_tfm_atten')
        self.kp_tfm_hidden = tf.placeholder(tf.float32, name='kp_tfm_hidden')

        self.cls_weight = tf.placeholder(tf.float32, (None,), name='cls_weight')

        self.computed_cls_weights = cls_weights if cls_weights is not None else [1 for _ in range(FLAGS.num_classes)]

        self.trainable_variables = None
        self.restore = restore

        self.logits, self.logits_adv, self.cost, self.cost_adv, self.cost_v_adv = self.construct_model()
        self.optimizer = self.build_optimizer(self.cost, adv=0)

        self.y_pred = tf.nn.softmax(self.logits, axis=1, name='y_pred')
        correct = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(self.y_pred, axis=1), name='correct')
        self.acc = tf.reduce_mean(tf.cast(correct, tf.float32), name='acc')

        self.y_pred_adv = self.y_pred
        self.acc_adv = self.acc

        if self.adv:
            self.y_pred_adv = tf.nn.softmax(self.logits_adv, axis=1, name='y_pred_adv')
            correct_adv = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(self.y_pred_adv, axis=1), name='correct_adv')
            self.acc_adv = tf.reduce_mean(tf.cast(correct_adv, tf.float32), name='acc_adv')

            self.optimizer_adv = self.build_optimizer(self.cost_adv, adv=1)

        # If writing information is desired in the future
        # tf.summary.scalar('cost', self.cost)
        # tf.summary.scalar('acc', self.acc)
        # self.merged_metrics = tf.summary.merge_all()
        # self.train_writer = tf.summary.FileWriter(os.path.join(FLAGS.tb_dir, 'train'))
        # self.val_writer = tf.summary.FileWriter(os.path.join(FLAGS.tb_dir, 'val'))

    @staticmethod
    def select_train_vars(print_stuff=True):
        train_vars = tf.trainable_variables()

        non_trainable_layers = ['/layer_{}/'.format(num)
                                for num in range(FLAGS.tfm_layers - FLAGS.tfm_ft_enc_layers)]
        if not FLAGS.tfm_ft_embed:
            non_trainable_layers.append('/word_embedding/' if FLAGS.tfm_type == 0 else '/embeddings/')
        if not FLAGS.tfm_ft_pooler:
            non_trainable_layers.append('/sequnece_summary/' if FLAGS.tfm_type == 0 else '/pooler/')

        train_vars = [v for v in train_vars if not any(z in v.name for z in non_trainable_layers)]

        if print_stuff:
            tf.logging.info('Removing: {}'.format(non_trainable_layers))
            tf.logging.info(train_vars)

        return train_vars

    def build_optimizer(self, cost_fn, adv):
        name = ('optimizer' if adv == 0 else ('optimizer_adv' if adv == 1 else 'optimizer_v_adv'))

        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(
            cost_fn, var_list=self.trainable_variables, name=name) if FLAGS.adam else tf.train.RMSPropOptimizer(
            learning_rate=FLAGS.lr).minimize(cost_fn, var_list=self.trainable_variables, name=name)

        return opt

    def construct_model(self):
        orig_embed, logits = self.fprop()
        self.trainable_variables = self.select_train_vars()
        loss = tf.identity(self.ce_loss(logits, self.cls_weight), name='cost')

        logits_adv, loss_adv = logits, loss

        if self.adv:
            logits_adv = self.fprop(orig_embed, loss, adv=True)
            loss_adv = tf.identity(FLAGS.adv_coeff * self.adv_loss(logits_adv, self.cls_weight), name='cost_adv')
            assert self.trainable_variables == self.select_train_vars(print_stuff=False)

        return logits, logits_adv, loss, loss_adv, None

    def fprop(self, orig_embed=None, reg_loss=None, adv=False):
        if adv: assert (reg_loss is not None and orig_embed is not None)

        nl_out = LanguageModel.build_xlnet_transformer_raw(
            self.x_nl[0], self.x_nl[1], self.x_nl[2], self.kp_tfm_atten, self.kp_tfm_hidden,
            adv, orig_embed, reg_loss, self.restore) if FLAGS.tfm_type == 0 else \
            LanguageModel.build_bert_transformer_raw(
                self.x_nl[0], self.x_nl[1], self.x_nl[2], self.kp_tfm_atten, self.kp_tfm_hidden,
                adv, orig_embed, reg_loss, self.restore)
        if not adv:
            orig_embed, nl_out = nl_out[0], nl_out[1]

        synth_out = tf.concat([nl_out, self.x_sent], axis=1)
        synth_out = tf.nn.dropout(synth_out, keep_prob=self.kp_cls)

        synth_out_shape = synth_out.get_shape()[1]

        with tf.variable_scope('fc_output/', reuse=tf.AUTO_REUSE):
            if FLAGS.cls_hidden > 0:
                hidden_weights = tf.get_variable('cb_hidden_weights', shape=(synth_out_shape, FLAGS.cls_hidden),
                                                 initializer=tf.contrib.layers.xavier_initializer())
                hidden_biases = tf.get_variable('cb_hidden_biases', shape=FLAGS.cls_hidden,
                                                initializer=tf.contrib.layers.xavier_initializer())

                synth_out = tf.matmul(synth_out, hidden_weights) + hidden_biases
                synth_out = tf.nn.dropout(synth_out, keep_prob=self.kp_cls)

            output_weights = tf.get_variable('cb_output_weights', shape=(
                FLAGS.cls_hidden if FLAGS.cls_hidden > 0 else synth_out_shape, FLAGS.num_classes),
                                             initializer=tf.contrib.layers.xavier_initializer())
            output_biases = tf.get_variable('cb_output_biases', shape=FLAGS.num_classes,
                                            initializer=tf.zeros_initializer())

            cb_out = tf.matmul(synth_out, output_weights) + output_biases

            return (orig_embed, cb_out) if not adv else cb_out

    @staticmethod
    def v_adv_loss(logits_reg, logits_perturb):
        return tf.distributions.kl_divergence(logits_reg, logits_perturb, name='v_adv_loss')

    def adv_loss(self, logits, cls_weight):
        return tf.identity(self.ce_loss(logits, cls_weight, adv=True), name='adv_loss')

    def ce_loss(self, logits, cls_weight, adv=False):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits)
        loss_l2 = 0

        if FLAGS.l2_reg_coeff > 0.0:  # and not adv: @TODO restore if necessary
            varlist = self.trainable_variables
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
            self.kp_tfm_atten: 1-0.1 if ver == 'train' else 1.0,
            self.kp_tfm_hidden: 1-0.1 if ver == 'train' else 1.0,
        }

        if batch_y is not None:
            feed_dict[self.y] = self.one_hot(batch_y)
            feed_dict[self.cls_weight] = self.get_cls_weights(batch_y)

        return feed_dict

    def train_neural_network(self, sess, batch_x, batch_y, adv):
        x_nl = [z[0] for z in batch_x]
        x_pos = [z[1] for z in batch_x]
        x_sent = [z[2] for z in batch_x]

        feed_dict = self.get_feed_dict(x_nl, x_pos, x_sent, batch_y, ver='train')

        sess.run(
            (self.optimizer if not adv else self.optimizer_adv),
            feed_dict=feed_dict
        )

    def execute_validation(self, sess, test_data, adv):
        n_batches = math.ceil(float(FLAGS.test_examples) / float(FLAGS.batch_size))
        val_loss, val_loss_adv, val_acc = 0.0, 0.0, 0.0
        tot_val_ex = 0

        all_y_pred = []
        all_y_pred_adv = []
        all_y = []
        for batch in range(n_batches):
            batch_x, batch_y = self.get_batch(batch, test_data, ver='validation')
            tloss, tloss_adv, tacc, _, tpred, tpred_adv = self.stats_from_run(sess, batch_x, batch_y, adv)

            val_loss += tloss
            val_acc += tacc * len(batch_y)
            tot_val_ex += len(batch_y)

            all_y_pred = np.concatenate((all_y_pred, tpred))
            all_y = np.concatenate((all_y, batch_y))

            if adv:
                val_loss_adv += tloss_adv
                all_y_pred_adv = np.concatenate((all_y_pred_adv, tpred_adv))

        val_loss /= tot_val_ex
        val_acc /= tot_val_ex
        val_f1, val_f1_adv = f1_score(all_y, all_y_pred, average='weighted'), None

        if adv:
            val_loss_adv /= tot_val_ex
            val_f1_adv = f1_score(all_y, all_y_pred_adv, average='weighted')

        return 'Dev Loss: {:>7.4f}{}Dev F1: {:>7.4f}{}'.format(val_loss, (
            ' Dev Adv Loss: {:>7.4f} '.format(val_loss_adv) if adv else ' '), val_f1, (
            ' Dev Adv F1: {:>7.4f} '.format(val_f1_adv) if adv else ''))

    def stats_from_run(self, sess, batch_x, batch_y, adv):
        x_nl = [z[0] for z in batch_x]
        x_pos = [z[1] for z in batch_x]
        x_sent = [z[2] for z in batch_x]

        feed_dict = self.get_feed_dict(x_nl, x_pos, x_sent, batch_y, ver='test')

        run_loss, run_acc, run_pred = sess.run([self.cost, self.acc, self.y_pred], feed_dict=feed_dict)
        run_pred = np.argmax(run_pred, axis=1)
        run_loss_adv, run_acc_adv, run_pred_adv = None, None, None

        if adv:
            run_loss_adv, run_acc_adv, run_pred_adv = sess.run([self.cost_adv, self.acc_adv, self.y_pred_adv],
                                                               feed_dict=feed_dict)
            run_pred_adv = np.argmax(run_pred_adv, axis=1)

        return np.sum(run_loss), np.sum(run_loss_adv), run_acc, run_acc_adv, run_pred, run_pred_adv

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
        epoch_string = str(epoch).zfill(3)

        if not os.path.isdir(FLAGS.cb_output_dir):
            os.mkdir(FLAGS.cb_output_dir)

        saver.save(sess, os.path.join(FLAGS.cb_output_dir + '/' + epoch_string + '/'), global_step=epoch)

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

    def load_model(self, default_graph, train=False):
        def get_last_save(scan_loc):
            ret_ar = []
            files = []
            directory = os.fsencode(scan_loc)

            for fstr in os.listdir(directory):
                if os.path.isdir(os.path.join(scan_loc, os.fsdecode(fstr))):
                    ret_ar.append(os.fsdecode(fstr))
                else:
                    files.append(os.fsdecode(fstr))

            if len(ret_ar) == 0 and len(files) > 0:
                return scan_loc + '/'

            ret_ar.sort()
            return os.path.join(scan_loc, ret_ar[-1]) + '/'

        def get_assignment_map_from_checkpoint(tvars):
            graph_var_names = [v.name[:-2] for v in tvars]
            assignment_map = collections.OrderedDict()

            for name in graph_var_names:
                assignment_map[name] = name

            return assignment_map, None

        dr = FLAGS.cb_output_dir if not train else FLAGS.cb_input_dir

        with default_graph.as_default():
            init_checkpoint = get_last_save(dr)
            am, _ = get_assignment_map_from_checkpoint(tf.trainable_variables())
            tf.train.init_from_checkpoint(init_checkpoint, am)
