from gensim.models import KeyedVectors

if __name__ == "__main__":
    model = KeyedVectors.load_word2vec_format("../data/glove/glove6b100d_gensim.txt", binary=False)

    print('model loaded')
    while True:
        word = input().strip('\t\r\n ')
        try:
            print(model[word])
        except:
            print('"{}" not in glove'.format(word))