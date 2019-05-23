def get_vocab_information(data):
    ret = {}

    for pair in data:
        words = pair[1].split(' ')
        for word in words:
            word = ''.join(ch for ch in word.printable if ch.isalnum())
            print(word)
            if word in ret:
                ret[word] += 1
            else:
                ret[word] = 1

    return sorted(ret.items(), key=lambda x: x[1], reverse=True)