import tensorflow as tf
import tensorflow_hub as hub

x = tf.placeholder(tf.string, (None, None))
x_len = tf.placeholder(tf.int32, (None,))

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)


tokens_input = [["the", "cat", "is", "on", "the", "mat"],
                ["dogs", "are", "in", "the", "fog", ""]]
tokens_length = [6, 5]
embeddings = elmo(
    inputs={
        "tokens": x,
        "sequence_len": x_len
    },
    signature="tokens",
    as_dict=True)["elmo"]

print(embeddings)