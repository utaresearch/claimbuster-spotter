from flags import FLAGS


def get_vocab_information(data):
    ret = {}
    #bk

    for pair in data:
        words = pair[1].split(' ')
        for word in words:
            if word in ret:
                ret[word] += 1
            else:
                ret[word] = 1

    return sorted(ret.items(), key=lambda x: x[1], reverse=True)


def get_embed_vocab_info():
    print("Loading embedding model | ", end='', flush=True)

    word_list = []

    with open(FLAGS.w2v_loc if FLAGS.embed_type == 0 else FLAGS.glove_loc, 'r') as f:
        print(f.readline(), end=' | ', flush=True)  # flush first line of metadata
        for line in f:
            split_line = line.split()
            word = split_line[0]
            word_list.append((word, 0))

    print("{} words loaded!".format(len(word_list)))

    return word_list
