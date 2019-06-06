from gensim.scripts.glove2word2vec import glove2word2vec


if __name__ == '__main__':
    glove2word2vec(glove_input_file="./glove840b.txt", word2vec_output_file="./glove840b_gensim.bin")
