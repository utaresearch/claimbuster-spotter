import numpy as np
import math
import os
from tqdm import tqdm
from utils.data_loader import DataLoader
from model import ClaimBusterModel
from sklearn.metrics import f1_score, classification_report
from flags import FLAGS
import tensorflow as tf


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(z) for z in FLAGS.gpu])

    tf.logging.info("Loading dataset")
    data_load = DataLoader()

    test_data = data_load.load_testing_data()
    tf.logging.info("{} testing examples".format(test_data.get_length()))

    graph = tf.get_default_graph()
    cb_model = ClaimBusterModel(data_load.vocab, data_load.class_weights, restore=True)
    cb_model.load_model(graph)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())

        n_batches = math.ceil(float(FLAGS.test_examples) / float(FLAGS.batch_size))
        n_samples = 0
        eval_loss, eval_acc = 0.0, 0.0

        y_all = []
        pred_all = []

        for i in tqdm(range(n_batches)):
            batch_x, batch_y = cb_model.get_batch(i, test_data, ver='test')

            b_loss, _, b_acc, _, b_pred, _ = cb_model.stats_from_run(sess, batch_x, batch_y, adv=False)
            if b_loss == 0 and b_acc == 0 and b_pred == 0:
                continue

            eval_loss += b_loss
            eval_acc += b_acc * len(batch_y)
            n_samples += len(batch_y)

            y_all = np.concatenate((y_all, batch_y))
            pred_all = np.concatenate((pred_all, b_pred))

        f1score = f1_score(y_all, pred_all, average='weighted')

        eval_loss /= n_samples
        eval_acc /= n_samples

        print('Labels:', y_all)
        print('Predic:', pred_all)

        tf.logging.info('Final stats | Loss: {:>7.4} Acc: {:>7.4f}% F1: {:>.4f}'.format(
            eval_loss, eval_acc * 100, f1score))

        target_names = (['NFS', 'UFS', 'CFS'] if FLAGS.num_classes == 3 else ['NFS/UFS', 'CFS'])
        print(classification_report(y_all, pred_all, target_names=target_names, digits=4))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
