#!/usr/bin/env bash

wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip

mv uncased_L-12_H-768_A-12 bert_pretrain_base
rm uncased_L-12_H-768_A-12.zip

wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

mv wwm_uncased_L-24_H-1024_A-16 bert_pretrain_large_wwm
rm wwm_uncased_L-24_H-1024_A-16.zip

wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
unzip uncased_L-24_H-1024_A-16.zip
mv uncased_L-24_H-1024_A-16 bert_pretrain_large
rm uncased_L-24_H-1024_A-16.zip