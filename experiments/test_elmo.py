import tensorflow
import tensorflow_hub as hub

if __name__ == '__main__':
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    tokens_input = [["the", "cat", "is", "on", "the", "mat"],
                    ["dogs", "are", "in", "the", "fog", ""]]
    tokens_length = [6, 5]
    embeddings = elmo(
        inputs={
            "tokens": tokens_input,
            "sequence_len": tokens_length
        },
        signature="tokens",
        as_dict=True)["elmo"]