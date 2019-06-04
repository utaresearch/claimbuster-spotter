import tensorflow as tf
import sys
sys.path.append('..')
from flags import FLAGS


def random_perturbation_loss(embedded, length, loss_fn):
    """Adds noise to embeddings and recomputes classification loss."""
    noise = tf.random_normal(shape=tf.shape(embedded))
    perturb = _scale_l2(_mask_by_length(noise, length), FLAGS.perturb_norm_length)
    return loss_fn(embedded + perturb)


def adversarial_loss(embedded, loss, loss_fn):
    """Adds gradient to embedding and recomputes classification loss."""
    grad, = tf.gradients(
        loss,
        embedded,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    grad = tf.stop_gradient(grad)
    perturb = _scale_l2(grad, FLAGS.perturb_norm_length)
    return loss_fn(embedded + perturb)


def _mask_by_length(t, length):
    """Mask t, 3-D [batch, time, dim], by length, 1-D [batch,]."""
    maxlen = t.get_shape().as_list()[1]

    # Subtract 1 from length to prevent the perturbation from going on 'eos'
    mask = tf.sequence_mask(length - 1, maxlen=maxlen)
    mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
    # shape(mask) = (batch, num_timesteps, 1)
    return t * mask


def _scale_l2(x, norm_length):
    # shape(x) = (batch, num_timesteps, d)
    # Divide x by max(abs(x)) for a numerically stable L2 norm.
    # 2norm(x) = a * 2norm(x/a)
    # Scale over the full sequence, dims (1, 2)
    alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
    l2_norm = alpha * tf.sqrt(tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
    x_unit = x / l2_norm
    return norm_length * x_unit
