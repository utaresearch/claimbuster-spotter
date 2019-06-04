from gensim.models import KeyedVectors
import numpy as np

if __name__ == "__main__":
    model = KeyedVectors.load_word2vec_format("../data/word2vec/GoogleNews-vectors-negative300.bin", binary=True)
    print(model["bosnia"])