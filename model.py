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
from optimizers.adam import AdamWeightFriction

K = tf.keras
L = K.layers


class ClaimBusterModel(K.models.Model):
    def __init__(self, cls_weights=None):
        super(ClaimBusterModel, self).__init__()
        self.layer = ClaimBusterLayer(cls_weights if cls_weights is not None else [1 for _ in range(FLAGS.num_classes)])
        self.adv = None

    def call(self, x_id, **kwargs):
        raise Exception('Please do not call this model. Use the *_on_batch functions instead')

    def warm_up(self):
        input_ph = K.layers.Input(shape=(FLAGS.max_len,), dtype='int32')
        self.layer.call(input_ph, training=False)

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

    def adv_train_on_batch(self, x_id, y):
        return self.layer.adv_train_on_batch(x_id, y)

    def stats_on_batch(self, x_id, y):
        return self.layer.stats_on_batch(x_id, y)

    def preds_on_batch(self, x_id):
        return self.layer.preds_on_batch(x_id)


class ClaimBusterLayer(K.layers.Layer):
    def __init__(self, cls_weights=None):
        super(ClaimBusterLayer, self).__init__()

        self.bert_model = LanguageModel.build_bert()
        self.dropout_layer = L.Dropout(rate=1-FLAGS.kp_cls)
        self.fc_layer = L.Dense(FLAGS.num_classes)

        self.computed_cls_weights = cls_weights

        self.optimizer = AdamWeightFriction(learning_rate=FLAGS.lr)
        self.vars_to_train = []

    def call(self, x_id, **kwargs):
        assert 'training' in kwargs

        training = kwargs.get('training')
        perturb = None if 'perturb' not in kwargs else kwargs.get('perturb')
        get_embedding = None if 'get_embedding' not in kwargs else kwargs.get('get_embedding')

        if not get_embedding:
            bert_output = self.bert_model(x_id, perturb, training=training)
        else:
            orig_embed, bert_output = self.bert_model(x_id, perturb, get_embedding=True, training=training)

        bert_output = self.dropout_layer(bert_output, training=training)
        ret = self.fc_layer(bert_output)

        if not self.vars_to_train:
            if not FLAGS.restore_and_continue:
                self.init_model_weights()
            self.vars_to_train = self.select_train_vars()

        if not get_embedding:
            return ret
        else:
            return orig_embed, ret

    @tf.function
    def train_on_batch(self, x_id, y):
        y = tf.one_hot(y, depth=FLAGS.num_classes)

        with tf.GradientTape() as tape:
            logits = self.call(x_id, training=True)
            loss = self.compute_training_loss(y, logits)

        grad = tape.gradient(loss, self.vars_to_train)
        self.optimizer.apply_gradients(zip(grad, self.vars_to_train))

        return tf.reduce_sum(loss), self.compute_accuracy(y, logits)

    @tf.function
    def adv_train_on_batch(self, x_id, y):
        y = tf.one_hot(y, depth=FLAGS.num_classes)

        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                orig_embed, logits = self.call(x_id, training=True, get_embedding=True)
                loss = self.compute_ce_loss(y, logits)

            perturb = self._compute_perturbation(loss, orig_embed, tape2)

            logits_adv = self.call(x_id, training=True, perturb=perturb)
            loss_adv = self.compute_training_loss(y, logits_adv)

        grad = tape.gradient(loss_adv, self.vars_to_train)
        self.optimizer.apply_gradients(zip(grad, self.vars_to_train))

        return tf.reduce_sum(loss_adv), self.compute_accuracy(y, logits_adv)

    @tf.function
    def stats_on_batch(self, x_id, y):
        y = tf.one_hot(y, depth=FLAGS.num_classes)
        logits = self.call(x_id, training=False)

        return tf.reduce_sum(self.compute_training_loss(y, logits)), self.compute_accuracy(y, logits)

    @tf.function
    def preds_on_batch(self, x_id):
        logits = self.call(x_id, training=False)
        return tf.nn.softmax(logits)

    def compute_training_loss(self, y, logits):
        loss = self.compute_ce_loss(y, logits)
        loss_l2 = 0

        if FLAGS.l2_reg_coeff > 0.0:
            varlist = self.vars_to_train
            loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in varlist if 'bias' not in v.name])

        if FLAGS.weight_classes_loss:
            print(loss)
            print(self.computed_cls_weights)
            print(y)

            y_int = tf.argmax(y, axis=1)
            print(y_int)
            # loss_adj = [self.computed_cls_weights[x] for x in y_int]
            cw = tf.constant(self.computed_cls_weights, dtype=tf.float32)
            print(cw)
            loss_adj = tf.map_fn(lambda x: cw[x], y_int, dtype=tf.float32)
            print(loss_adj)

            loss *= loss_adj

        return loss + FLAGS.l2_reg_coeff * loss_l2

    @staticmethod
    def compute_ce_loss(y, logits):
        return tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)

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

    @staticmethod
    def _compute_perturbation(loss, orig_embed, tape):
        grad = tape.gradient(loss, orig_embed)
        grad = tf.stop_gradient(grad)
        perturb = FLAGS.perturb_norm_length * grad / tf.norm(grad, ord='euclidean')

        return perturb

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

            if ns[1] not in ["encoder", "embeddings", "pooler"]:
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
            if ns[1] == "pooler":
                return "/".join(ns)
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
