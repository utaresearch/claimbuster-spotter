import tensorflow as tf
import tensorflow_hub as hub

x = tf.placeholder(tf.string, (None, ))

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
# elmo_output = elmo(
#     ["the cat is on the mat", "dogs are in the fog"],
#     signature="default",
#     as_dict=True)

elmo_output = elmo(
    x,
    signature="default",
    as_dict=True)

print(elmo_output)