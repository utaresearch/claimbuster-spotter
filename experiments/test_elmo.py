import tensorflow as tf
import tensorflow_hub as hub

pl = tf.placeholder(tf.string, (None, 200))

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

tokens_input = [["the", "cat", "is", "on", "the", "mat"],
                ["dogs", "are", "in", "the", "fog", ""]]
tokens_length = [200 for _ in tokens_input]
elmo_output = elmo(
    inputs={"tokens": pl, "sequence_len": tokens_length},
    signature="tokens",
    as_dict=True)

print(elmo_output)