import numpy as np
import pandas as pd
import time
from core.api.api_wrapper import ClaimSpotterAPI

if __name__ == '__main__':
    api = ClaimSpotterAPI()

    print('--- 2020 SOTU Processing ---')

    sotus = [pd.read_csv('data/sotu_2020.csv'), pd.read_csv('data/sotu_2019.csv')]
    df = pd.concat(sotus)
    df = df[['score', 'text']]
    df = df.dropna()

    print(df)

    gstart = time.time()
    logits = api.batch_sentence_query(df['text'].to_list())
    df['logits_ncs'] = [x[0] for x in logits]
    df['logits_cfs'] = [x[1] for x in logits]
    dur = time.time() - gstart

    print(df)

    print('Completed processing in {} sec'.format(dur))
    df.to_csv('logit_analysis_output.csv')
