import numpy as np
import pandas as pd
from core.api.api_wrapper import ClaimSpotterAPI

if __name__ == '__main__':
    api = ClaimSpotterAPI()

    print('--- 2020 SOTU Processing ---')

    sotu = pd.read_csv('data/sotu_2020.csv')
    sotu = sotu[['score', 'text']]
    sotu = sotu.dropna()

    print(sotu)

    sotu['logits'] = [x[1] for x in api.batch_sentence_query(sotu['text'].to_list())]

    sotu.to_csv('logit_analysis_output.csv')
