import numpy as np
import nltk.data
import re
import matplotlib.pyplot as plt
import requests
from urllib.parse import quote
import argparse

from api_wrapper import ClaimBusterAPI


def main():
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    fp = open("./data/ClintonTrump.txt")
    rawdata = fp.read()
    data = rawdata.split('\n')
    data = [x for x in data if not x == '']

    sentence_list = []
    for x in data:
        interm = [re.sub(r'[A-Z0-9][A-Z0-9. ]*: ', '', x) for x in tokenizer.tokenize(x)]
        if len(sentence_list) == 0:
            sentence_list = interm
        else:
            sentence_list = np.concatenate([sentence_list, interm])

    # Get claimbuster information
    if args.advcb:
        api = ClaimBusterAPI()
        scores_advcb = sorted([api.direct_sentence_query(x)[2] for x in sentence_list])
        print('\n'.join([str(x) for x in scores_advcb]))

    if args.svmcb:
        scores_svmcb = [requests.get("https://idir.uta.edu/factchecker/score_text/{}".format(quote(x))).json()["results"][0]["score"] for x in sentence_list]
        print('\n'.join([str(x) for x in scores_advcb]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-advcb', True)
    parser.add_argument('-svmcb', True)
    args = parser.parse_args()

    main()