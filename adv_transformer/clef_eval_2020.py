import pandas as pd
import csv
from adv_transformer.core.api.api_wrapper import ClaimSpotterAPI


def main():
    api = ClaimSpotterAPI()

    df = pd.read_csv('./data/clef20/test-input.tsv', delimiter='\t')
    res = [x[1] for x in api.batch_sentence_query(df['tweet_text'].tolist())]
    df['score'] = ['{:f}'.format(x) for x in res]
    df['run_id'] = [-1 for _ in range(len(res))]
    df.drop(['tweet_url', 'tweet_text'], axis=1, inplace=True)
    df.to_csv('../clef-out.tsv', index=False, sep='\t', header=False, quoting=csv.QUOTE_NONE)


if __name__ == '__main__':
    main()
