from gensim.models import KeyedVectors

if __name__ == "__main__":
    model = KeyedVectors.load_word2vec_format("../data/word2vec/GoogleNews-vectors-negative300.bin", binary=True)
    print(model["smart"])
    print(model.most_similar(positive=['smart']))
    print(model.most_similar(positive=['Trump']))
    print(model.most_similar(positive=['Fruit']))
    print(model.most_similar(positive=['Apple']))
    print(model["dog"])