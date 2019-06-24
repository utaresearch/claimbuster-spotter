import tensorflow as tf
import numpy as np
import os
from model import ClaimBusterModel
from utils.data_loader import DataLoader
from utils import transformations as transf
from flags import FLAGS

return_strings = ['Non-factual statement', 'Unimportant factual statement', 'Salient factual statement']


def prc_sentence(sentence, vocab):
    def get_idx(str):
        if str in vocab:
            return vocab[str]
        else:
            return 0

    sent = transf.get_sentiment(sentence)
    pos = transf.process_sentence_full_tags(sentence)
    sentence = [get_idx(z) for z in transf.transform_sentence_complete(sentence).split(' ')]

    return sentence, pos, sent


def subscribe_query(sess, cb_model, vocab):
    print('Enter a sentence to process')
    sentence_tuple = prc_sentence(input().strip('\n\r\t '), vocab)
    return cb_model.get_preds(sess, sentence_tuple)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(z) for z in FLAGS.gpu_active])

    data_load = DataLoader()
    vocab = data_load.vocab

    transf.load_dependencies()
    cb_model = ClaimBusterModel(restore=True)

    graph = tf.Graph()
    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        cb_model.load_model(sess, graph)

        while True:
            res = subscribe_query(sess, cb_model, vocab)
            idx = np.argmax(res, axis=1)

            print(res)

            print('{} with probability {}'.format(np.array(return_strings)[idx][0], res[0][idx][0]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
