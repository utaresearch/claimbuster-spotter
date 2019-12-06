import sys
import os
import tensorflow as tf

sys.path.append('..')
from flags import FLAGS


def get_adversarial_perturbation(embedded, loss):
    grad, = tf.gradients(loss, embedded, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    grad = tf.stop_gradient(grad)
    perturb = _scale_perturb(grad, FLAGS.perturb_norm_length)
    return perturb


def _scale_perturb(x, norm_length):
    return norm_length * x / tf.norm(x, ord='euclidean')
