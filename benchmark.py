import numpy as np
import string
import random
from core.api.api_wrapper import ClaimSpotterAPI


def generate_sentence():
    return ''.join([random.choice(string.ascii_letters) for _ in range(140)])


if __name__ == '__main__':
    api = ClaimSpotterAPI()

    sentence_list = [generate_sentence() for _ in range(10000)]
    print(api.batch_sentence_query(sentence_list))
    print('Test OK')
