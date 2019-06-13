from keras.preprocessing.text import text_to_word_sequence


if __name__ == '__main__':
    # define the document
    text = 'the koala-lion was hungry and ate an APPLE.'
    # tokenize the document
    result = text_to_word_sequence(text)
    print(result)