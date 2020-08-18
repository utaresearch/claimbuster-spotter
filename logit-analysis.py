import numpy as np
import pandas as pd
import time
from core.api.api_wrapper import ClaimSpotterAPI

if __name__ == '__main__':
    api = ClaimSpotterAPI()

    print('--- 2020 SOTU Processing ---')

    sotu = pd.read_csv('data/sotu_2020.csv')
    sotu = sotu[['score', 'text']]
    sotu = sotu.dropna()

    print(sotu)

    gstart = time.time()
    sotu['logits'] = [x[1] for x in api.batch_sentence_query(sotu['text'].to_list())]
    dur = time.time() - gstart

    print(sotu)

    print('Completed processing in {} sec'.format(dur))
    sotu.to_csv('logit_analysis_output.csv')
