import numpy as np
from api_wrapper import ClaimBusterAPI

if __name__ == '__main__':
    api = ClaimBusterAPI()

    print('--- Batch Sentence Query ---')

    sentence_list = [
        'Donald Trump is the 45th President of the United States',
        'I really like cheese',
        'McDonalds earns $10 billion dollars each minute'
    ]

    print(sentence_list)
    print(api.batch_sentence_query())
    print('Test OK')

    print('--- Single Sentence Query ---')

    print(sentence_list[0])
    print(api.single_sentence_query(sentence_list[0]))
    print('Test OK')

    print('--- Command Line Query ---')

    while True:
        res = api.subscribe_cmdline_query()[0]
        idx = np.argmax(res)

        print(res)
        print('{} with probability {}'.format(api.return_strings[idx], res[idx]))
