#!/usr/bin/env bash

wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
mv glove.840B.300d.txt ./glove840b.txt
rm -r glove.840B.300d.zip
python3 glove_to_w2v.py --glove_inp=./glove840b.txt --w2v_out=./glove840b_gensim.txt

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mv glove.6B.50d.txt ./glove6b50d.txt
mv glove.6B.100d.txt ./glove6b100d.txt
mv glove.6B.200d.txt ./glove6b200d.txt
mv glove.6B.300d.txt ./glove6b300d.txt
rm -r glove.6B.zip
python3 glove_to_w2v.py --glove_inp=./glove6b50d.txt --w2v_out=./glove6b50d_gensim.txt
python3 glove_to_w2v.py --glove_inp=./glove6b100d.txt --w2v_out=./glove6b100d_gensim.txt
python3 glove_to_w2v.py --glove_inp=./glove6b200d.txt --w2v_out=./glove6b200d_gensim.txt
python3 glove_to_w2v.py --glove_inp=./glove6b300d.txt --w2v_out=./glove6b300d_gensim.txt