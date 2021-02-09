import pandas as pd
import csv
import os
from adv_transformer.core.api.api_wrapper import ClaimSpotterAPI


def main():
    api = ClaimSpotterAPI()

    loc = 'data/clef20/task5/raw'
    for el in os.listdir(loc):
        fpath = os.path.join(loc, el)
        df = pd.read_csv(fpath, delimiter='\t', names=['num', 'speaker', 'text', 'label'])

        res = [x[1] for x in api.batch_sentence_query(df['text'].tolist())]
        df['score'] = ['{:f}'.format(x) for x in res]
        df[['num', 'score']].to_csv(f'../clef2020_task5_{el}', index=False, sep='\t', header=False)


if __name__ == '__main__':
    main()
