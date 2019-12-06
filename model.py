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

K = tf.keras
L = K.layers


class ClaimBusterModel(K.layers.Layer):
    def __init__(self, cls_weights=None):
        super(ClaimBusterModel, self).__init__()

        self.optimizer = K.optimizers.Adam(learning_rate=FLAGS.lr)
        self.accuracy = K.metrics.Accuracy()  # @TODO wtf create some more shit?
        self.computed_cls_weights = cls_weights if cls_weights is not None else [1 for _ in range(FLAGS.num_classes)]

    def call(self,
             x_id, x_mask, x_segment,  # BERT inputs
             nl_len, nl_output_mask,  # Not sure? @TODO figure out
             y,  # Ground truths
             kp_cls, kp_tfm_atten, kp_tfm_hidden,  # Dropout parameters
             cls_weight):

        bert_output = LanguageModel.build_bert(x_id)
        bert_output = tf.nn.dropout(bert_output, rate=1-FLAGS.kp_cls)

        fc_layer = K.Dense(FLAGS.num_classes)
        ret = fc_layer(bert_output)

        self.select_train_vars()
        return ret

    @tf.function
    def train_on_batch(self, x_id, x_mask, x_segment, y):
        with tf.GradientTape() as tape:
            logits = self.call(x_id, x_mask, x_segment, -1, -1, y, FLAGS.kp_cls, FLAGS.kp_tfm_atten, FLAGS.kp_tfm_hidden)
            loss = self.compute_loss(y, logits)

        grad = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.trainable_weights))

        return loss

        # self.accuracy.update_state(y, yhat)  # @TODO update accuracy

    def compute_loss(self, y, logits):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
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
        print(train_vars)
        exit()

        non_trainable_layers = ['/layer_{}/'.format(num)
                                for num in range(FLAGS.tfm_layers - FLAGS.tfm_ft_enc_layers)]
        if not FLAGS.tfm_ft_embed:
            non_trainable_layers.append('/word_embedding/' if FLAGS.tfm_type == 0 else '/embeddings/')
        if not FLAGS.tfm_ft_pooler:
            non_trainable_layers.append('/sequnece_summary/' if FLAGS.tfm_type == 0 else '/pooler/')

        train_vars = [v for v in train_vars if not any(z in v.name for z in non_trainable_layers)]

        logging.info('Removing: {}'.format(non_trainable_layers))
        logging.info(train_vars)

        raise Exception('cmon you didn\'t update self.trainable_vars')
