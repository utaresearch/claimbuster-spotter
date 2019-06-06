#!/usr/bin/env bash

python3 -m spacy download en_core_web_lg

cd data/word2vec
./get_w2v.sh

cd ../glove
./get_glove.sh