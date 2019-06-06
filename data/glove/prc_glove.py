import numpy as np
import pickle


def loadGloveModel(gloveFile):
    print("Loading Glove Model")

    vec_list = []
    word_list = []

    with open(gloveFile, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            vec_list.append(embedding)
            word_list.append(word)

    print("{} words loaded!".format(len(word_list)))

