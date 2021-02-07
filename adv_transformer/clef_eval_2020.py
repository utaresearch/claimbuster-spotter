import pandas as pd
from adv_transformer.core.api.api_wrapper import ClaimSpotterAPI


def main():
    api = ClaimSpotterAPI()

    df = pd.read_csv('./data/clef20/test-input.tsv', delimiter='\t')
    res = api.batch_sentence_query(df['tweet_text'].tolist())
    df['score'] = res
    df.drop(['tweet_url', 'tweet_text'])
    df.to_frame().to_csv('../clef-out', sep='\t')


if __name__ == '__main__':
    main()
