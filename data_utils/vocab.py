import pickle


def get_vocab_information(data_loc):
    with open(data_loc, 'rb') as f:
        data = pickle.load(f)

    ret = {}

    for pair in data:
        words = pair[1].split(' ')
        for word in words:
            if word in ret:
                ret[word] += 1
            else:
                ret[word] = 1

    return sorted(ret.items(), key=lambda x: x[1], reverse=True)