import numpy as np
import string
import random
import time
from core.api.api_wrapper import ClaimSpotterAPI


def generate_sentence():
    return ''.join([random.choice(string.ascii_letters) for _ in range(140)])


if __name__ == '__main__':
    api = ClaimSpotterAPI()

    sentence_list = [generate_sentence() for _ in range(10000)]

    gstart = time.time()
    api.batch_sentence_query(sentence_list)
    gend = time.time()

    print('{} tweets processed in {} seconds: {} tweets per second'.format(len(sentence_list), gend - gstart, len(sentence_list) / (gend - gstart)))
    print('Test OK')
