import math
import time
import os
from utils.data_loader import DataLoader
from model import ClaimBusterModel
from flags import FLAGS, print_flags
import tensorflow as tf


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(z) for z in FLAGS.gpu])

    print_flags()

    tf.logging.info("Loading dataset")
    data_load = DataLoader()

    train_data = data_load.load_training_data()
    test_data = data_load.load_testing_data()

    tf.logging.info("{} training examples".format(train_data.get_length()))
    tf.logging.info("{} validation examples".format(test_data.get_length()))

    print(tf.train.list_variables('{}-0'.format(FLAGS.cb_output_dir)))
    exit()

    cb_model = ClaimBusterModel(data_load.vocab, data_load.class_weights, restore=True, adv=True)

    graph = tf.get_default_graph()
    cb_model.load_model(graph, train=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True, allow_growth=True)) as sess:

        start = time.time()
        epochs_trav = 0

        tf.logging.info("Starting adv training...")
        for epoch in range(FLAGS.advtrain_steps):
            epochs_trav += 1
            n_batches = math.ceil(float(FLAGS.train_examples) / float(FLAGS.batch_size))

            n_samples = 0
            epoch_loss, epoch_loss_adv, epoch_acc, epoch_acc_adv = 0.0, 0.0, 0.0, 0.0

            for i in range(n_batches):
                batch_x, batch_y = cb_model.get_batch(i, train_data)
                cb_model.train_neural_network(sess, batch_x, batch_y, adv=True)

                b_loss, b_loss_adv, b_acc, b_acc_adv, _, _ = cb_model.stats_from_run(sess, batch_x, batch_y, adv=True)
                epoch_loss += b_loss
                epoch_loss_adv += b_loss_adv
                epoch_acc += b_acc * len(batch_y)
                epoch_acc_adv += b_acc_adv * len(batch_y)
                n_samples += len(batch_y)

            epoch_loss /= n_samples
            epoch_loss_adv /= n_samples
            epoch_acc /= n_samples
            epoch_acc_adv /= n_samples

            if epoch % FLAGS.stat_print_interval == 0:
                log_string = 'Epoch {:>3} Loss: {:>7.4} Adv Loss: {:>7.4} Acc: {:>7.4f}% Adv Acc: {:>7.4f}% '.format(
                    epoch + 1, epoch_loss, epoch_loss_adv, epoch_acc * 100, epoch_acc_adv * 100)
                if test_data.get_length() > 0:
                    log_string += cb_model.execute_validation(sess, test_data, adv=True)
                log_string += '({:3.3f} sec/epoch)'.format((time.time() - start) / epochs_trav)

                tf.logging.info(log_string)

                start = time.time()
                epochs_trav = 0

            if epoch % FLAGS.model_save_interval == 0 and epoch != 0:
                cb_model.save_model(sess, epoch)
                tf.logging.info('Model @ epoch {} saved'.format(epoch + 1))

        tf.logging.info('Training complete. Saving final model...')
        cb_model.save_model(sess, FLAGS.advtrain_steps)
        tf.logging.info('Model saved.')

        sess.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
