#!/usr/bin/env bash

wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
gunzip GoogleNews-vectors-negative300.bin.gz
python3 w2v_to_txt.py