import tensorflow as tf
import numpy as np
import math
import os
from data_utils.data_loader import DataLoader
from flags import FLAGS

x = tf.placeholder(tf.int32, (None, FLAGS.max_len), name='x')
x_len = tf.placeholder(tf.int32, (None,), name='x_len')
output_mask = tf.placeholder(tf.bool, (None, FLAGS.max_len), name='output_mask')
y = tf.placeholder(tf.int32, (None, FLAGS.num_classes), name='y')
kp_emb = tf.placeholder(tf.float32, name='kp_emb')
kp_lstm = tf.placeholder(tf.float32, name='kp_lstm')


def pad_seq(inp):
    ret = np.full((len(inp), FLAGS.max_len), -1, dtype=np.int32)
    for i in range(len(inp)):
        ret[i][:len(inp[i])] = inp[i]
    return ret


def one_hot(a):
    return np.squeeze(np.eye(FLAGS.num_classes)[np.array(a).reshape(-1)])


def eval_stats(sess, batch_x, batch_y, cost, acc):
    eval_loss = sess.run(
        cost,
        feed_dict={
            x: pad_seq(batch_x),
            x_len: [len(el) for el in batch_x],
            output_mask: [[1 if j == len(el) - 1 else 0 for j in range(FLAGS.max_len)] for el in batch_x],
            y: one_hot(batch_y),
            kp_emb: 1.0,
            kp_lstm: 1.0
        }
    )
    eval_acc = sess.run(
        acc,
        feed_dict={
            x: pad_seq(batch_x),
            x_len: [len(el) for el in batch_x],
            output_mask: [[1 if j == len(el) - 1 else 0 for j in range(FLAGS.max_len)] for el in batch_x],
            y: one_hot(batch_y),
            kp_emb: 1.0,
            kp_lstm: 1.0
        }
    )

    return np.sum(eval_loss), eval_acc


def load_model(sess, graph):
    def get_last_save(scan_loc):
        ret_ar = []
        directory = os.fsencode(scan_loc)
        for fstr in os.listdir(directory):
            if '.meta' in os.fsdecode(fstr):
                ret_ar.append(os.fsdecode(fstr))
        ret_ar.sort()
        return ret_ar[-1]

    model_dir = '{}/{}'.format(FLAGS.save_loc, get_last_save(FLAGS.save_loc))
    tf.logging.info('Attempting to restore from {}'.format(model_dir))

    with graph.as_default():
        saver = tf.train.import_meta_graph(model_dir)
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.save_loc))

        # outputs
        cost = graph.get_tensor_by_name('cost:0')
        y_pred = graph.get_tensor_by_name('y_pred:0')
        acc = graph.get_tensor_by_name('acc:0')

        tf.logging.info('Model successfully restored.')
        return cost, y_pred, acc


def get_batch(bid, data):
    batch_x = []
    batch_y = []

    for i in range(FLAGS.batch_size):
        idx = bid * FLAGS.batch_size + i
        if idx >= FLAGS.testing_examples:
            break
        batch_x.append(data.x[idx])
        batch_y.append(data.y[idx])

    return batch_x, batch_y


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    tf.logging.info("Loading dataset")
    data_load = DataLoader()

    test_data = data_load.load_testing_data()
    tf.logging.info("{} testing examples".format(test_data.get_length()))

    graph = tf.Graph()
    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        cost, y_pred, acc = load_model(sess, graph)

        n_batches = math.ceil(float(FLAGS.train_examples) / float(FLAGS.batch_size))

        n_samples = 0
        eval_loss = 0.0
        eval_acc = 0.0

        for i in range(n_batches):
            batch_x, batch_y = get_batch(i, test_data)

            b_loss, b_acc = eval_stats(sess, batch_x, batch_y, cost, acc)
            eval_loss += b_loss
            eval_acc += b_acc * len(batch_y)
            n_samples += len(batch_y)

        eval_acc /= n_samples

        tf.logging.info('Final stats | Loss: {:>7.4} Acc: {:>7.4f}% '.format(eval_loss, eval_acc * 100))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
