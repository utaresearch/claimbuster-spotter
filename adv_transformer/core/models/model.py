# Copyright (C) 2020 IDIR Lab - UT Arlington
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License v3 as published by
#     the Free Software Foundation.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact Information:
#     See: https://idir.uta.edu/cli.html
#
#     Chengkai Li
#     Box 19015
#     Arlington, TX 76019
#
import os
import re
import tensorflow as tf
import json
from adv_transformer.core.utils.flags import FLAGS
from absl import logging
from transformers import TFAutoModel, AutoConfig


class ClaimSpotterModel(tf.keras.models.Model):
    def __init__(self, cls_weights=None):
        super(ClaimSpotterModel, self).__init__()
        self.layer = ClaimSpotterLayer(cls_weights if cls_weights is not None else [1 for _ in range(FLAGS.cs_num_classes)])
        self.adv = None

    def call(self, x, **kwargs):
        raise Exception('Do not call this model. Use the *_on_batch functions instead')

    def warm_up(self):
        input_ph_id = tf.keras.layers.Input(shape=(FLAGS.cs_max_len,), dtype='int32')
        input_ph_sent = tf.keras.layers.Input(shape=(2,), dtype='float32')

        self.layer.call((input_ph_id, input_ph_sent), training=False)

    def load_custom_model(self, loc=None):
        model_dir = (loc if loc is not None else FLAGS.cs_model_dir)

        if any('.ckpt' in x for x in os.listdir(model_dir)):
            load_location = model_dir
        else:
            folders = [x for x in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, x))]
            load_location = os.path.join(model_dir, sorted(folders)[-1])

        last_epoch = int(load_location.split('/')[-1].split('_')[-1])
        load_location = os.path.join(load_location, FLAGS.cs_model_ckpt)

        logging.info('Retrieving pre-trained weights from {}'.format(load_location))
        self.load_weights(load_location)

        return last_epoch

    def save_custom_model(self, epoch, fold, metrics_obj):
        loc = os.path.join(FLAGS.cs_model_dir, 'fold_{}_{}'.format(str(fold + 1).zfill(2), str(epoch + 1).zfill(3)))
        self.save_weights(os.path.join(loc, FLAGS.cs_model_ckpt))
        with open(os.path.join(loc, 'val_res.json'), 'w') as f:
            json.dump(metrics_obj, f)

        return loc

    def train_on_batch(self, x, y):
        return self.layer.train_on_batch(x, y)

    def adv_train_on_batch(self, x, y):
        return self.layer.adv_train_on_batch(x, y)

    def stats_on_batch(self, x, y):
        return self.layer.stats_on_batch(x, y)

    def preds_on_batch(self, x):
        return self.layer.preds_on_batch(x)


class ClaimSpotterLayer(tf.keras.layers.Layer):
    def __init__(self, cls_weights=None):
        super(ClaimSpotterLayer, self).__init__()

        config = AutoConfig.from_pretrained(FLAGS.cs_tfm_type)
        config.hidden_dropout_prob = 1 - FLAGS.cs_kp_tfm_hidden
        config.attention_probs_dropout_prob = 1 - FLAGS.cs_kp_tfm_atten
        self.bert_model = TFAutoModel.from_pretrained(config)

        self.dropout_layer = tf.keras.layers.Dropout(rate=1-FLAGS.cs_kp_cls)
        self.fc_output_layer = tf.keras.layers.Dense(FLAGS.cs_num_classes)
        if FLAGS.cs_cls_hidden:
            self.fc_hidden_layer = tf.keras.layers.Dense(FLAGS.cs_cls_hidden, activation='relu')

        self.computed_cls_weights = cls_weights

        self.optimizer = tf.optimizers.Adam(learning_rate=FLAGS.cs_lr)
        self.vars_to_train = []

    def call(self, x, **kwargs):
        assert 'training' in kwargs

        x_id = [x[0], tf.zeros((tf.shape(x[0])), dtype=tf.int32)]
        x_sent = x[1]

        training = kwargs.get('training')
        perturb = kwargs.get('perturb', None)
        get_embedding = kwargs.get('get_embedding', -1)

        if get_embedding == -1:
            bert_output = self.bert_model(x_id, training=training, perturb=perturb)
        else:
            orig_embed, bert_output = self.bert_model(x_id, perturb=perturb, get_embedding=get_embedding, training=training)

        bert_output = tf.concat([bert_output, x_sent], axis=1)
        bert_output = self.dropout_layer(bert_output, training=training)

        if FLAGS.cs_cls_hidden:
            bert_output = self.fc_hidden_layer(bert_output)
            bert_output = self.dropout_layer(bert_output, training=training)

        ret = self.fc_output_layer(bert_output)

        if not self.vars_to_train:
            if not FLAGS.cs_restore_and_continue:
                self.init_model_weights()
            self.vars_to_train = self.select_train_vars()

        if get_embedding == -1:
            return ret
        else:
            return orig_embed, ret

    @tf.function
    def train_on_batch(self, x, y):
        y = tf.one_hot(y, depth=FLAGS.cs_num_classes)

        with tf.GradientTape() as tape:
            logits = self.call(x, training=True)
            loss = self.compute_training_loss(y, logits)

        grad = tape.gradient(loss, self.vars_to_train)
        self.optimizer.apply_gradients(zip(grad, self.vars_to_train))

        return tf.reduce_sum(loss), self.compute_accuracy(y, logits)

    @tf.function
    def adv_train_on_batch(self, x, y):
        y = tf.one_hot(y, depth=FLAGS.cs_num_classes)

        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                orig_embed, logits = self.call(x, training=True, get_embedding=FLAGS.cs_perturb_id)
                loss = self.compute_ce_loss(y, logits)

            perturb = self._compute_perturbation(loss, orig_embed, tape2)

            logits_adv = self.call(x, training=True, perturb=perturb)
            loss_adv = self.compute_training_loss(y, logits_adv) + FLAGS.cs_lambda * loss

        grad = tape.gradient(loss_adv, self.vars_to_train)
        self.optimizer.apply_gradients(zip(grad, self.vars_to_train))

        return tf.reduce_sum(loss_adv), self.compute_accuracy(y, logits_adv)

    @tf.function
    def stats_on_batch(self, x, y):
        y = tf.one_hot(y, depth=FLAGS.cs_num_classes)
        logits = self.call(x, training=False)

        return tf.reduce_sum(self.compute_training_loss(y, logits)), self.compute_accuracy(y, logits)

    @tf.function
    def preds_on_batch(self, x):
        logits = self.call(x, training=False)
        return tf.nn.softmax(logits)

    def compute_training_loss(self, y, logits):
        loss = self.compute_ce_loss(y, logits)
        loss_l2 = 0

        if FLAGS.cs_l2_reg_coeff > 0.0:
            varlist = self.vars_to_train
            loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in varlist if 'bias' not in v.name])

        if FLAGS.cs_weight_classes_loss:
            y_int = tf.argmax(y, axis=1)
            cw = tf.constant(self.computed_cls_weights, dtype=tf.float32)
            loss_adj = tf.map_fn(lambda x: cw[x], y_int, dtype=tf.float32)

            loss *= loss_adj

        return loss + FLAGS.cs_l2_reg_coeff * loss_l2

    @staticmethod
    def compute_ce_loss(y, logits):
        return tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)

    def select_train_vars(self):
        train_vars = self.trainable_variables

        if FLAGS.cs_tfm_type == 'albert' and FLAGS.cs_tfm_ft_enc_layers == 0:
            non_trainable_layers = ['/encoder/']
        else:
            non_trainable_layers = ['/layer_{}/'.format(num)
                                    for num in range(FLAGS.cs_tfm_layers - FLAGS.cs_tfm_ft_enc_layers)]
        if not FLAGS.cs_tfm_ft_embed:
            non_trainable_layers.append('/embeddings/')
        if not FLAGS.cs_tfm_ft_pooler:
            non_trainable_layers.append('/pooler/')

        train_vars = [v for v in train_vars if not any(z in v.name for z in non_trainable_layers)]

        logging.info('Removing: {}'.format(non_trainable_layers))
        logging.info('Trainable variables: {}'.format([v.name for v in train_vars]))

        return train_vars

    @staticmethod
    def compute_accuracy(y, logits):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1)), dtype=tf.float32))

    @staticmethod
    def _compute_perturbation(loss, orig_embed, tape):
        grad = tape.gradient(loss, orig_embed)
        grad = tf.stop_gradient(grad)
        perturb = FLAGS.cs_perturb_norm_length * grad / tf.norm(grad, ord='euclidean')

        return perturb

    def init_model_weights(self):
        if FLAGS.cs_tfm_type == 'bert':
            self.load_bert_weights()
        else:
            self.load_albert_weights()
