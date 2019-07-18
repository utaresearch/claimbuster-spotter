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

    if os.path.isdir(FLAGS.tb_dir):
        os.rmdir(FLAGS.tb_dir)

    tf.logging.info("Loading dataset")
    data_load = DataLoader()

    train_data = data_load.load_training_data()
    test_data = data_load.load_testing_data()

    tf.logging.info("{} training examples".format(train_data.get_length()))
    tf.logging.info("{} validation examples".format(test_data.get_length()))

    cb_model = ClaimBusterModel(data_load.vocab, data_load.class_weights, restore=FLAGS.restore_and_continue, adv=False)

    graph = tf.get_default_graph()
    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True, allow_growth=True)) as sess:
        if not FLAGS.restore_and_continue:
            if not FLAGS.use_bert_hub:
                tf.logging.info('Restoring pretrained BERT weights into graph')
            sess.run(tf.global_variables_initializer())
        else:
            cb_model.load_model(sess, graph, train=True)

        start = time.time()
        epochs_trav = 0

        tf.logging.info("Starting training...")
        for epoch in range(FLAGS.pretrain_steps):
            epochs_trav += 1
            n_batches = math.ceil(float(FLAGS.train_examples) / float(FLAGS.batch_size))

            n_samples = 0
            epoch_loss, epoch_acc = 0.0, 0.0

            for i in range(n_batches):
                batch_x, batch_y = cb_model.get_batch(i, train_data)
                cb_model.train_neural_network(sess, batch_x, batch_y, adv=False)

                b_loss, _, b_acc, _, _, _ = cb_model.stats_from_run(sess, batch_x, batch_y, adv=False)
                epoch_loss += b_loss
                epoch_acc += b_acc * len(batch_y)
                n_samples += len(batch_y)

            epoch_loss /= n_samples
            epoch_acc /= n_samples

            if epoch % FLAGS.stat_print_interval == 0:
                log_string = 'Epoch {:>3} Loss: {:>7.4} Acc: {:>7.4f}% '.format(epoch + 1, epoch_loss, epoch_acc * 100)
                if test_data.get_length() > 0:
                    log_string += cb_model.execute_validation(sess, test_data, adv=False)
                log_string += '({:3.3f} sec/epoch)'.format((time.time() - start) / epochs_trav)

                tf.logging.info(log_string)

                start = time.time()
                epochs_trav = 0

            if epoch % FLAGS.model_save_interval == 0 and epoch != 0:
                cb_model.save_model(sess, epoch)
                tf.logging.info('Model @ epoch {} saved'.format(epoch + 1))

        tf.logging.info('Training complete. Saving final model...')
        cb_model.save_model(sess, FLAGS.pretrain_steps)
        tf.logging.info('Model saved.')

        sess.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
