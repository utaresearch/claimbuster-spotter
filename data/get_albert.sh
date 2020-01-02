#!/usr/bin/env bash

wget https://storage.googleapis.com/albert_models/albert_base_v2.tar.gz

tar -xvzf albert_base_v2.tar.gz
rm albert_base_v2.tar.gz
mv albert_base albert_pretrain_base

wget https://storage.googleapis.com/albert_models/albert_large_v2.tar.gz
tar -xvzf albert_large_v2.tar.gz
rm albert_large_v2.tar.gz
mv albert_large albert_pretrain_large