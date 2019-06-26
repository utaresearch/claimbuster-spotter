import tensorflow as tf
from bert import run_classifier
import numpy as np
import os
from model import ClaimBusterModel
from utils.data_loader import DataLoader
from utils import transformations as transf
from flags import FLAGS

return_strings = ['Non-factual statement', 'Unimportant factual statement', 'Salient factual statement']
tokenizer = DataLoader.create_tokenizer_from_hub_module()


def extract_info(sentence, vocab):
    def get_idx(str):
        if str in vocab:
            return vocab[str]
        else:
            return 0

    sentence = transf.transform_sentence_complete(sentence)

    sent = transf.get_sentiment(sentence)
    pos = transf.process_sentence_full_tags(sentence)
    
    if not FLAGS.bert_model:
        sentence = [get_idx(z) for z in sentence.split(' ')]

    return sentence, pos, sent


def prc_sentence(sentence, vocab):
    global tokenizer

    sentence, pos, sent = extract_info(sentence, vocab)

    if FLAGS.bert_model:
        input_examples = [run_classifier.InputExample(guid="", text_a=sentence, text_b=None, label=0)]
        input_features = run_classifier.convert_examples_to_features(input_examples,
                                                                     [z for z in range(FLAGS.num_classes)],
                                                                     FLAGS.max_len, tokenizer)
    else:
        input_features = sentence

    return input_features, pos, sent


def subscribe_query(sess, cb_model, vocab):
    print('Enter a sentence to process')
    sentence_tuple = prc_sentence(input().strip('\n\r\t '), vocab)
    return cb_model.get_preds(sess, sentence_tuple)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(z) for z in FLAGS.gpu])

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
    tf.logging.set_verbosity(tf.logging.ERROR)
    main()
