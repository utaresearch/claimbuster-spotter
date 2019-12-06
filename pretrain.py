import math
import time
import os
from shutil import rmtree
from utils.data_loader import DataLoader
from model import ClaimBusterModel
from flags import FLAGS, print_flags
from absl import logging
import tensorflow as tf
import numpy as np
from model import ClaimBusterModel

K = tf.keras


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(z) for z in FLAGS.gpu])

    print_flags()

    if os.path.isdir(FLAGS.tb_dir):
        rmtree(FLAGS.tb_dir)
    if os.path.isdir(FLAGS.cb_model_dir) and not FLAGS.restore_and_continue:
        print('Continue? (y/n) You will overwrite the contents of FLAGS.cb_model_dir ({})'.format(FLAGS.cb_model_dir))
        inp = input().strip('\r\n\t')
        if inp.lower() == 'y':
            rmtree(FLAGS.cb_model_dir)
        else:
            print('Exiting...')
            exit()
    if not os.path.isdir(FLAGS.cb_model_dir) and FLAGS.restore_and_continue:
        raise Exception('Cannot restore from non-existent folder: {}'.format(FLAGS.cb_model_dir))

    logging.info("Loading dataset")
    data_load = DataLoader()

    train_data = data_load.load_training_data()
    test_data = data_load.load_testing_data()

    logging.info("{} training examples".format(train_data.get_length()))
    logging.info("{} validation examples".format(test_data.get_length()))

    model = ClaimBusterModel()
    dataset = tf.data.Dataset.from_tensor_slices(([x[0] for x in train_data.x], train_data.y)).shuffle(
        buffer_size=train_data.get_length()).batch(FLAGS.batch_size)

    logging.info("Starting training...")

    input = K.layers.Input(shape=(None, FLAGS.max_len))
    kmodel = K.models.Model(inputs=input, outputs=model(input))

    kmodel.compile(optimizer=model.optimizer, loss=tf.keras.losses.CategoricalCrossentropy())
    print(kmodel.summary())
    kmodel.fit(dataset, epochs=FLAGS.pretrain_steps)

    epochs_trav = 0
    for epoch in range(FLAGS.pretrain_steps):
        epochs_trav += 1
        epoch_loss, epoch_acc = 0, 0
        start = time.time()

        for x_id, y in dataset:
            epoch_loss += np.sum(model.train_on_batch(x_id, y))

        if epoch % FLAGS.stat_print_interval == 0:
            log_string = 'Epoch {:>3} Loss: {:>7.4} Acc: {:>7.4f}% '.format(epoch + 1, epoch_loss, epoch_acc * 100)
            log_string += '({:3.3f} sec/epoch)'.format((time.time() - start) / epochs_trav)

            logging.info(log_string)

            start = time.time()
            epochs_trav = 0


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    main()
