import tensorflow as tf

a = tf.Variable([1, 2, 3, 4])
len_pl = tf.placeholder(tf.int32)

slop = a[len_pl]


if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        res = sess.run(
            slop,
            feed_dict={
                len_pl: 3
            }
        )

        print(res)