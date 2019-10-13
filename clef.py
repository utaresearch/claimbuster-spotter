import pandas as pd
from api_wrapper import ClaimBusterAPI

api = ClaimBusterAPI()


def get_score(text):
    api_result = api.direct_sentence_query(text)
    return api_result[-1]


df = pd.read_csv("data/CT19-T1-Test.csv")

df['score'] = df.apply(lambda x: get_score(x['text']), axis=1)

df.to_csv("clef_out.csv", index=None)