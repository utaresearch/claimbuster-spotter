import sys
import os
import tensorflow as tf

cwd = os.getcwd()
root_dir = None

for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.endswith("ac_bert.txt"):
            root_dir = root

if cwd != root_dir:
    from ..flags import FLAGS
else:
    sys.path.append('..')
    from flags import FLAGS


def get_adversarial_perturbation(embedded, loss):
    grad, = tf.gradients(loss, embedded, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    grad = tf.stop_gradient(grad)
    perturb = _scale_perturb(grad, FLAGS.perturb_norm_length)
    return perturb


def _scale_perturb(x, norm_length):
    return norm_length * x / tf.norm(x, ord='euclidean')
