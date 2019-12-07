import numpy as np
import collections
import os
import math
import re
import tensorflow as tf
from sklearn.metrics import f1_score
from flags import FLAGS
from absl import logging
from models.lang_model import LanguageModel
from models.bert2 import load_stock_weights

K = tf.keras
L = K.layers


class ClaimBusterModel(K.models.Model):
    def __init__(self, training, cls_weights=None):
        super(ClaimBusterModel, self).__init__()
        self.layer = ClaimBusterLayer(training)
        self.computed_cls_weights = cls_weights if cls_weights is not None else [1 for _ in range(FLAGS.num_classes)]

    def call(self, x_id):
        return self.layer.call(x_id)

    def warm_up(self):
        input_ph = K.layers.Input(shape=(FLAGS.max_len,), dtype='int32')
        self.call(input_ph)

    def load_custom_model(self):
        if any('.ckpt' in x for x in os.listdir(FLAGS.cb_model_dir)):
            load_location = FLAGS.cb_model_dir
        else:
            folders = [x for x in os.listdir(FLAGS.cb_model_dir) if os.path.isdir(os.path.join(FLAGS.cb_model_dir, x))]
            load_location = os.path.join(FLAGS.cb_model_dir, sorted(folders)[-1])

        last_epoch = int(load_location.split('/')[-1])
        load_location = os.path.join(load_location, FLAGS.cb_model_ckpt)
        self.load_weights(load_location)

        return last_epoch

    def save_custom_model(self, epoch):
        self.save_weights(os.path.join(FLAGS.cb_model_dir, str(epoch + 1).zfill(3), FLAGS.cb_model_ckpt))

    def train_on_batch(self, x_id, y):
        return self.layer.train_on_batch(x_id, y)

    def stats_on_batch(self, x_id, y):
        return self.layer.stats_on_batch(x_id, y)

    def preds_on_batch(self, x_id):
        return self.layer.preds_on_batch(x_id)


class ClaimBusterLayer(K.layers.Layer):
    def __init__(self, training):
        super(ClaimBusterLayer, self).__init__()

        self.bert_model = LanguageModel.build_bert()
        self.dropout_layer = L.Dropout(rate=1-FLAGS.kp_cls)
        self.fc_layer = L.Dense(FLAGS.num_classes)

        self.optimizer = K.optimizers.Adam(learning_rate=FLAGS.lr)
        self.vars_to_train = []
        self.is_training = training

    def call(self, x_id):
        bert_output = self.bert_model(x_id, training=self.is_training)
        bert_output = self.dropout_layer(bert_output, training=self.is_training)
        ret = self.fc_layer(bert_output)

        if not self.vars_to_train:
            if not FLAGS.restore_and_continue:
                self.init_model_weights()
            self.vars_to_train = self.select_train_vars()

        return ret

    @tf.function
    def train_on_batch(self, x_id, y):
        y = tf.one_hot(y, depth=FLAGS.num_classes)

        with tf.GradientTape() as tape:
            logits = self.call(x_id)
            loss = self.compute_loss(y, logits)

        grad = tape.gradient(loss, self.vars_to_train)
        self.optimizer.apply_gradients(zip(grad, self.vars_to_train))

        return tf.reduce_sum(loss), self.compute_accuracy(y, logits)

    @tf.function
    def stats_on_batch(self, x_id, y):
        y = tf.one_hot(y, depth=FLAGS.num_classes)
        logits = self.call(x_id)

        return tf.reduce_sum(self.compute_loss(y, logits)), self.compute_accuracy(y, logits)

    @tf.function
    def preds_on_batch(self, x_id):
        logits = self.call(x_id)
        return tf.nn.softmax(logits)

    def compute_loss(self, y, logits):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss_l2 = 0

        if FLAGS.l2_reg_coeff > 0.0:
            varlist = self.trainable_variables
            loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in varlist if 'bias' not in v.name]) * FLAGS.l2_reg_coeff

        ret_loss = loss + loss_l2

        if FLAGS.weight_classes_loss:
            ret_loss *= self.computed_cls_weights

        return tf.identity(ret_loss, name='loss')

    def select_train_vars(self):
        train_vars = self.trainable_variables

        non_trainable_layers = ['/layer_{}/'.format(num)
                                for num in range(FLAGS.tfm_layers - FLAGS.tfm_ft_enc_layers)]
        if not FLAGS.tfm_ft_embed:
            non_trainable_layers.append('/word_embedding/' if FLAGS.tfm_type == 0 else '/embeddings/')
        if not FLAGS.tfm_ft_pooler:
            non_trainable_layers.append('/sequnece_summary/' if FLAGS.tfm_type == 0 else '/pooler/')

        train_vars = [v for v in train_vars if not any(z in v.name for z in non_trainable_layers)]

        logging.info('Removing: {}'.format(non_trainable_layers))
        logging.info([v.name for v in train_vars])

        return train_vars

    @staticmethod
    def compute_accuracy(y, logits):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1)), dtype=tf.float32))

    def init_model_weights(self, ckpt_path=os.path.join(FLAGS.bert_model_loc, 'bert_model.ckpt')):
        # Define several helper functions
        def bert_prefix():
            re_bert = re.compile(r'(.*)/(embeddings|encoder)/(.+):0')
            match = re_bert.match(self.weights[0].name)
            assert match, "Unexpected bert layer: {} weight:{}".format(self, self.weights[0].name)
            prefix = match.group(1)
            return prefix

        def map_to_stock_variable_name(name, pfx="bert"):
            name = name.split(":")[0]
            ns = name.split("/")
            pns = pfx.split("/")

            if ns[:len(pns)] != pns:
                return None

            name = "/".join(["bert"] + ns[len(pns):])
            ns = name.split("/")

            if ns[1] not in ["encoder", "embeddings"]:
                return None
            if ns[1] == "embeddings":
                if ns[2] == "LayerNorm":
                    return name
                elif ns[2] == "word_embeddings_projector":
                    ns[2] = "word_embeddings_2"
                    if ns[3] == "projector":
                        ns[3] = "embeddings"
                        return "/".join(ns[:-1])
                    return "/".join(ns)
                else:
                    return "/".join(ns[:-1])
            if ns[1] == "encoder":
                if ns[3] == "intermediate":
                    return "/".join(ns[:4] + ["dense"] + ns[4:])
                else:
                    return name
            return None

        ckpt_reader = tf.train.load_checkpoint(ckpt_path)

        stock_weights = set(ckpt_reader.get_variable_to_dtype_map().keys())
        prefix = bert_prefix()

        loaded_weights = set()
        skip_count = 0
        weight_value_tuples = []
        skipped_weight_value_tuples = []

        bert_params = self.weights
        param_values = K.backend.batch_get_value(self.weights)

        for ndx, (param_value, param) in enumerate(zip(param_values, bert_params)):
            stock_name = map_to_stock_variable_name(param.name, prefix)

            if stock_name and ckpt_reader.has_tensor(stock_name):
                ckpt_value = ckpt_reader.get_tensor(stock_name)

                if param_value.shape != ckpt_value.shape:
                    logging.info("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                                 "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                            stock_name, ckpt_value.shape))
                    skipped_weight_value_tuples.append((param, ckpt_value))
                    continue

                weight_value_tuples.append((param, ckpt_value))
                loaded_weights.add(stock_name)
            else:
                print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param.name, stock_name, ckpt_path))
                skip_count += 1
        K.backend.batch_set_value(weight_value_tuples)

        logging.info("Done loading {} BERT weights from: {} into {} (prefix:{}). "
                     "Count of weights not found in the checkpoint was: [{}]. "
                     "Count of weights with mismatched shape: [{}]".format(
            len(weight_value_tuples), ckpt_path, self, prefix, skip_count, len(skipped_weight_value_tuples)))

        logging.info("Unused weights from checkpoint: {}".format("\n\t" + "\n\t".join(
            sorted(stock_weights.difference(loaded_weights)))))

        return skipped_weight_value_tuples
