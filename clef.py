import pandas as pd
from api.api_wrapper import ClaimSpotterAPI

api = ClaimSpotterAPI()


def get_score(text):
    api_result = api.single_sentence_query(text)
    return api_result[-1]


df = pd.read_csv("data/CT19-T1-Test.csv")

df['score'] = df.apply(lambda x: get_score(x['text']), axis=1)
df.to_csv("clef_out.csv", index=None)