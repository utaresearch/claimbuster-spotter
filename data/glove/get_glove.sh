#!/usr/bin/env bash

wget http://nlp.stanford.edu/data/glove.840B.300d.zip
gunzip unzip glove.840B.300d.zip
mv glove.840B.300d/glove.840B.300d.txt ./glove840b.txt
rm -r glove.840B.300d
python3 glove_to_w2v.py