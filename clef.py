import pandas as pd
from api_wrapper import ClaimBusterAPI

api = ClaimBusterAPI()


def get_score(text):
    api_result = api.direct_sentence_query(text)
    return api_result[-1]


df = pd.read_csv("data/CT19-T1-Test.csv")

df['score'] = df.apply(lambda x: get_score(x['text']), axis=1)

df.to_csv("clef_out.csv", columns=['line_number', 'score'], header=False, index=None)

fout = open('clef_gen.csv', 'w')

with open('clef_out.csv', 'r') as fin:
    for line in fin:
        data = line.split(',')
        data[1] = str(round(float(data[1]), 4))
        fout.write('\t'.join(data) + '\n')

import os
os.remove('clef_out.csv')
fout.close()
