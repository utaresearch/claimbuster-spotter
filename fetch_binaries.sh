#!/usr/bin/env bash

python3 -m spacy download en_core_web_lg
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz \
    -O data/word2vec/GoogleNews-vectors-negative300.bin.gz
gunzip data/word2vec/GoogleNews-vectors-negative300.bin.gz