#!/usr/bin/env bash

wget https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip
unzip cased_L-12_H-768_A-12.zip

mv xlnet_cased_L-12_H-768_A-12 xlnet_pretrain
rm cased_L-12_H-768_A-12.zip

cd xlnet_pretrain
mkdir model
mv xlnet_model.ckpt.data-00000-of-00001 model
mv xlnet_model.ckpt.index model
mv xlnet_model.ckpt.meta model

wget https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip
unzip xlnet_cased_L-24_H-1024_A-16.zip

mv xlnet_cased_L-24_H-1024_A-16 xlnet_pretrain_large
rm xlnet_cased_L-24_H-1024_A-16.zip

cd xlnet_pretrain_large
mkdir model
mv xlnet_model.ckpt.data-00000-of-00001 model
mv xlnet_model.ckpt.index model
mv xlnet_model.ckpt.meta model