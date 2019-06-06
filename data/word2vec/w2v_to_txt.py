from gensim.models.keyedvectors import KeyedVectors


if __name__ == '__main__':
    model = KeyedVectors.load_word2vec_format('./w2v3b_gensim.bin', binary=True)
    model.save_word2vec_format('./w2v3b_gensim.txt', binary=False)