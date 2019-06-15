from gensim.scripts.glove2word2vec import glove2word2vec
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove_inp')
    parser.add_argument('--w2v_out')

    glove2word2vec(glove_input_file="./glove840b.txt", word2vec_output_file="./glove840b_gensim.txt")


from gensim.scripts.glove2word2vec import glove2word2vec
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove_inp')
    parser.add_argument('--w2v_out')
    args = parser.parse_args()

    glove2word2vec(glove_input_file=args.glove_inp, word2vec_output_file=args.glove_out)
