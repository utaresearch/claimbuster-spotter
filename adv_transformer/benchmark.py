import numpy as np
import string
import random
import time
from adv_transformer.core.api.api_wrapper import ClaimSpotterAPI
from adv_transformer.core.utils.flags import FLAGS


def generate_sentence():
    return ''.join([random.choice(string.ascii_letters) for _ in range(140)])


if __name__ == '__main__':
    api = ClaimSpotterAPI()

    sentence_list = [generate_sentence() for _ in range(5000)]

    print('### single query test ###')

    gstart = time.time()
    for el in sentence_list:
        api.single_sentence_query(el)
    gend = time.time()

    print('single query | {} tweets processed in {} seconds | {} tweets per second'.format(
        len(sentence_list), gend - gstart, len(sentence_list) / (gend - gstart)))

    print('### batch query test ###')

    test_batches = [32, 64, 128]
    for btc in test_batches:
        FLAGS.cs_batch_size_reg = btc

        gstart = time.time()
        api.batch_sentence_query(sentence_list)
        gend = time.time()

        print('batch size {} | {} tweets processed in {} seconds | {} tweets per second'.format(
            FLAGS.cs_batch_size_reg, len(sentence_list), gend - gstart, len(sentence_list) / (gend - gstart)))

    print('Test OK')
