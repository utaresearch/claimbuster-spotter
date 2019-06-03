import tensorflow as tf
from models.recurrent import RecurrentModel
from flags import FLAGS

x = tf.placeholder(tf.int32, (None, FLAGS.max_len), name='x')
x_len = tf.placeholder(tf.int32, (None,), name='x_len')
y = tf.placeholder(tf.int32, (None, FLAGS.num_classes), name='y')
kp_emb = tf.placeholder(tf.float32, name='kp_emb')
kp_lstm = tf.placeholder(tf.float32, name='kp_lstm')


if __name__ == '__main__':
    embed = tf.get_variable('dummy_embed', shape=(100, FLAGS.embedding_dims))

    lstm_model = RecurrentModel()
    logits, cost = lstm_model.construct_model(x, x_len, y, embed, kp_emb, kp_lstm)
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)

    y_pred = tf.nn.softmax(logits, axis=1, name='y_pred')
    correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32), name='acc')