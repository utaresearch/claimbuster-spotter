import tensorflow as tf
import tensorflow_hub as hub

pl = tf.placeholder(tf.int32, (None, 200, 1))

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
# elmo_output = elmo(
#     ["the cat is on the mat", "dogs are in the fog"],
#     signature="default",
#     as_dict=True)

elmo_output = elmo(
    pl,
    signature="default",
    as_dict=True)

print(elmo_output)