import tensorflow as tf
import math
import time
import os
import numpy as np
from sklearn.metrics import f1_score, classification_report
from adv_bert_claimspotter.utils.data_loader import Dataset, DataLoader
from adv_bert_claimspotter.model import ClaimBusterModel
from adv_bert_claimspotter.flags import FLAGS, print_flags


label_mapping = {
    'nfs': -1,
    'ufs': 0,
    'cfs': 1
}


def map_label(n):
    return label_mapping[n]


def train_adv_bert_model(train, dev, test):
    global label_mapping

    tf.logging.set_verbosity(tf.logging.INFO)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(z) for z in FLAGS.gpu])

    print_flags()

    tf.logging.info("Loading dataset from given values")

    train = list(zip(train[0], list(map(map_label, train[1]))))
    dev = list(zip(dev[0], list(map(map_label, dev[1]))))
    test = list(zip(test[0], list(map(map_label, test[1]))))

    data_load = DataLoader(train, test, dev)

    train_data = data_load.load_training_data()
    test_data = data_load.load_testing_data()

    tf.logging.info("{} training examples".format(train_data.get_length()))
    tf.logging.info("{} validation examples".format(test_data.get_length()))

    cb_model = ClaimBusterModel(data_load.vocab, data_load.class_weights)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        if not FLAGS.bert_model:
            cb_model.embed_obj.init_embeddings(sess)

        start = time.time()
        epochs_trav = 0

        tf.logging.info("Starting{}training...".format(' adversarial ' if FLAGS.adv_train else ' '))
        for epoch in range(FLAGS.max_steps):
            epochs_trav += 1
            n_batches = math.ceil(float(FLAGS.train_examples) / float(FLAGS.batch_size))

            n_samples = 0
            epoch_loss = 0.0
            epoch_acc = 0.0

            # for i in range(n_batches):
            for i in range(1):
                batch_x, batch_y = cb_model.get_batch(i, train_data)
                cb_model.train_neural_network(sess, batch_x, batch_y)

                b_loss, b_acc, _ = cb_model.stats_from_run(sess, batch_x, batch_y)
                epoch_loss += b_loss
                epoch_acc += b_acc * len(batch_y)
                n_samples += len(batch_y)

            epoch_loss /= n_samples
            epoch_acc /= n_samples

            if epoch % FLAGS.stat_print_interval == 0:
                log_string = 'Epoch {:>3} Loss: {:>7.4} Acc: {:>7.4f}% '.format(epoch + 1, epoch_loss,
                                                                                epoch_acc * 100)
                if test_data.get_length() > 0:
                    log_string += cb_model.execute_validation(sess, test_data)
                log_string += '({:3.3f} sec/epoch)'.format((time.time() - start) / epochs_trav)

                tf.logging.info(log_string)

                start = time.time()
                epochs_trav = 0

            if epoch % FLAGS.model_save_interval == 0 and epoch != 0:
                cb_model.save_model(sess, epoch)
                tf.logging.info('Model @ epoch {} saved'.format(epoch + 1))

        tf.logging.info('Training complete. Saving final model...')
        cb_model.save_model(sess, FLAGS.max_steps)
        tf.logging.info('Model saved.')

    test_adv_bert_model(train, dev, test)


def test_adv_bert_model(train, dev, test):
    tf.logging.info('Evaluating model...')

    tf.logging.info("Loading dataset")
    data_load = DataLoader(train, dev, test)

    test_data = data_load.load_testing_data()
    tf.logging.info("{} testing examples".format(test_data.get_length()))

    cb_model = ClaimBusterModel(data_load.vocab, data_load.class_weights, restore=True)

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