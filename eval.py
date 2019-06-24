import tensorflow as tf
import numpy as np
import math
import os
from utils.data_loader import DataLoader
from model import ClaimBusterModel
from sklearn.metrics import f1_score, classification_report
from flags import FLAGS


def main():
    global computed_cls_weights

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(z) for z in FLAGS.gpu_active])

    cb_model = ClaimBusterModel(restore=True)

    graph = tf.Graph()
    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        cb_model.load_model(sess, graph)

        n_batches = math.ceil(float(FLAGS.test_examples) / float(FLAGS.batch_size))

        n_samples = 0
        eval_loss = 0.0
        eval_acc = 0.0

        y_all = []
        pred_all = []

        for i in range(n_batches):
            batch_x, batch_y = cb_model.get_batch(i, test_data, ver='test')

            b_loss, b_acc, b_pred = cb_model.stats_from_run(sess, batch_x, batch_y)
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
        print(classification_report(y_all, pred_all, target_names=target_names))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
