import tensorflow as tf
import numpy as np
import pickle
import os
from utils import transformations as transf
from flags import FLAGS

x = tf.placeholder(tf.int32, (None, FLAGS.max_len), name='x')
x_len = tf.placeholder(tf.int32, (None,), name='x_len')
output_mask = tf.placeholder(tf.bool, (None, FLAGS.max_len), name='output_mask')
y = tf.placeholder(tf.int32, (None, FLAGS.num_classes), name='y')
kp_emb = tf.placeholder(tf.float32, name='kp_emb')
kp_lstm = tf.placeholder(tf.float32, name='kp_lstm')
ext_vocab = []
return_strings = ['Non-factual statement', 'Unimportant factual statement', 'Salient factual statement']


def pad_seq(inp):
    ret = np.full((len(inp), FLAGS.max_len), -1, dtype=np.int32)
    for i in range(len(inp)):
        ret[i][:len(inp[i])] = inp[i]
    return ret


def one_hot(a):
    return np.squeeze(np.eye(FLAGS.num_classes)[np.array(a).reshape(-1)])


def load_model(sess, graph):
    global x, x_len, output_mask, y, kp_emb, kp_lstm

    def get_last_save(scan_loc):
        ret_ar = []
        directory = os.fsencode(scan_loc)
        for fstr in os.listdir(directory):
            if '.meta' in os.fsdecode(fstr) and 'cb.ckpt-' in os.fsdecode(fstr):
                ret_ar.append(os.fsdecode(fstr))
        ret_ar.sort()
        return ret_ar[-1]

    model_dir = os.path.join(FLAGS.model_dir, get_last_save(FLAGS.model_dir))
    tf.logging.info('Attempting to restore from {}'.format(model_dir))

    with graph.as_default():
        saver = tf.train.import_meta_graph(model_dir)
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))

        # inputs
        x = graph.get_tensor_by_name('x:0')
        x_len = graph.get_tensor_by_name('x_len:0')
        output_mask = graph.get_tensor_by_name('output_mask:0')
        y = graph.get_tensor_by_name('y:0')
        kp_emb = graph.get_tensor_by_name('kp_emb:0')
        kp_lstm = graph.get_tensor_by_name('kp_lstm:0')

        # outputs
        y_pred = graph.get_tensor_by_name('y_pred:0')

        tf.logging.info('Model successfully restored.')

        return y_pred


def get_batch(bid, data):
    batch_x = []
    batch_y = []

    for i in range(FLAGS.batch_size):
        idx = bid * FLAGS.batch_size + i
        if idx >= (FLAGS.total_examples if FLAGS.disjoint_data else FLAGS.test_examples):
            break
        batch_x.append(data.x[idx])
        batch_y.append(data.y[idx])

    return batch_x, batch_y


def load_ext_vocab():
    global ext_vocab

    with open(FLAGS.vocab_loc, 'rb') as f:
        ext_vocab = [z[0] for z in pickle.load(f)]


def parse_sentence(sentence):
    def vocab_idx(ch):
        global ext_vocab

        try:
            return ext_vocab.index(ch)
        except:
            return -1

    # sentence = transf.transform_sentence_complete(sentence)
    return [vocab_idx(word) for word in sentence.split(' ')]


def subscribe_query(sess, y_pred):
    print('Enter a sentence to process')
    sentence = parse_sentence(input().strip('\n\r\t '))
    batch_x = [sentence]
    preds = sess.run(
        y_pred,
        feed_dict={
            x: pad_seq(batch_x),
            x_len: [len(el) for el in batch_x],
            output_mask: [[1 if j == len(el) - 1 else 0 for j in range(FLAGS.max_len)] for el in batch_x],
            kp_emb: 1.0,
            kp_lstm: 1.0
        }
    )
    return preds


def main():
    global return_strings

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    load_ext_vocab()
    # transf.load_dependencies()

    graph = tf.Graph()
    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        y_pred = load_model(sess, graph)

        res = subscribe_query(sess, y_pred)
        idx = np.argmax(res, axis=1)

        print('{} with probability {}'.format(np.array(return_strings)[idx][0], res[0][idx][0]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
