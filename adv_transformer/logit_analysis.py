import numpy as np
import pandas as pd
import time
from adv_transformer.core.api.api_wrapper import ClaimSpotterAPI

if __name__ == '__main__':
    api = ClaimSpotterAPI()

    print('--- 2020 SOTU Processing ---')

    # dfs = [pd.read_csv('data/sotu_2020.csv'), pd.read_csv('data/sotu_2019.csv'), pd.read_csv('data/debates_2020.csv')]

    dfs = [pd.read_csv('data/sotu_2020.csv')]
    # dfs = [pd.read_csv('data/sotu_2019.csv')]
    df = pd.concat(dfs)
    df = df[['text']]
    df = df.dropna()

    print(df)

    gstart = time.time()
    logits = api.batch_sentence_query(df['text'].to_list())
    df['logits_ncs'] = [x[0] for x in logits]
    df['logits_cfs'] = [x[1] for x in logits]
    dur = time.time() - gstart

    print(df)

    print('Completed processing in {} sec'.format(dur))
    print('{} claims / sec'.format(len(df) / dur))
    df.to_csv('logit_analysis_output.csv')
