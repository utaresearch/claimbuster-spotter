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


def _mask_by_length(t, length):
    """Mask t, 3-D [batch, time, dim], by length, 1-D [batch,]."""
    maxlen = t.get_shape().as_list()[1]

    # Subtract 1 from length to prevent the perturbation from going on 'eos'
    mask = tf.sequence_mask(length - 1, maxlen=maxlen)
    mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
    # shape(mask) = (batch, num_timesteps, 1)
    return t * mask


def _scale_perturb(x, norm_length):
    return norm_length * x / tf.norm(x, ord='euclidean')
