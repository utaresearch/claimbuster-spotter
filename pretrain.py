import tensorflow as tf
import os
from data_utils.data_loader import DataLoader
from models.bdlstm import RecurrentModel
from flags import FLAGS

x = tf.placeholder(tf.int32, (None, None), name='x')
y = tf.placeholder(tf.int32, (None,), name='y')


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    tf.logging.info("Loading dataset")
    data_load = DataLoader()

    train_data = data_load.load_training_data()
    validation_data = data_load.load_validation_data()

    tf.logging.info("{} training examples".format(train_data.get_length()))
    tf.logging.info("{} validation examples".format(validation_data.get_length()))

    lstm_model = RecurrentModel()
    logits, cost = lstm_model.construct_model(x, y)
    print(logits, cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)


if __name__ == '__main__':
    main()
