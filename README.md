# Adversarial Claim Spotting
In this repository, we apply adversarial training as as regularization technique for the purpose of determining whether a given claim is a) factual and b) worthy of fact checking.

**WARNING! This documentation is deprecated. Updates are coming soon.**

## Requirements

Will be listed in the future; for now, please simply use `idir-server10` to run the code. Necessary packages should already be installed; otherwise, please follow the stack track and custom-install if needed.

## Experimental Setup

* Intel(R) Xeon(R) CPU E5-2695 v4 @ 2.10GHz
* 256GB RAM
* 4x Nvidia GTX 1080Ti (12GB RAM each)

## Code Overview

This section provides a high-level overview of this repository, as well as details regarding the functions of each source file.

### Pre-processing

Preprocessing is accomplished by [`pretrain.py`](pretrain.py). Sentences are extracted from `./data/`, transformed based on various flags defined in [`flags.py`](flags.py), and converted into the token/segment/mask format required by BERT. Uses `tensorflow-hub` module.

### Training

Each training session is predicated upon a pre-trained BERT model as an initialization point. After loading these weights, we apply two stages of training:

* Classifier Fine-Tuning for 5 epochs: [`pretrain.py`](pretrain.py)
* Adversarial Classifier Fine-Tuning for 50 epochs: [`advtrain.py`](advtrain.py)

Depending on the VRAM capacity of the selected GPU, as well as the predefined batch size, training time can range between 1 and 10 hours. On UTA servers, it should take approximately 6-7 hours.

### Evaluations

We output F1 scores on the `disjoint_2000.json` dataset using [`eval.py`](eval.py). In the future, a full corpus of quotes from a recent presidential campaign debate series will be used to quantify the in-the-wild reliability of the claim-spotting model.

### Interactive ClaimBuster Demo

**Warning: This file is deprecated! Please do not run until it is fixed in a future push and this warning message is removed!**

Using [`demo.py`](demo.py), users can input individual sentences into the command line, and the model will produce an inference result on the inputted sentence. This process should take under 500ms.

### Command-Line Flags

All flags are defined and editable in [`flags.py`](flags.py).

<!-- ## Different Modes for Executing Code

In [`utils/`](utils/), there are 3 files with varying purposes, as listed below. [`run_eval.sh`](utils/run_eval.sh) is used for official evaluation.
* [`run_sm.sh`](utils/run_sm.sh): Uses [small dataset](data/data_small.json) split into training/testing
* [`run_lg.sh`](utils/run_lg.sh): Uses [large dataset](data/data_large.json) split into training/testing
* [`run_eval.sh`](utils/run_eval.sh): Uses the entire [small dataset](data/data_small.json) for training and the [2000 pre-selected disjoint sentences](data/disjoint_2000.pkl) for evaluation. These are the commands described below in the example procedure. -->

## End-to-End Claim Spotting Procedure

### Clone GitHub repository
```bash
git clone https://github.com/kmeng01/adversarial-claimspotting.git
```

### CD to current directory

From this point forwards, all directories are referenced relative to the project
root.

```bash
cd adversarial-claimspotting
```

### Fetch word2vec and spaCy models

Because word2vec and spaCy binaries, which are required for pre-processing, are inconvenient/impossible to track with Git, they must be downloaded at time of use. There is a convenient pre-written script for this purpose.

```bash
chmod +x ./dependencies.sh
./dependencies.sh
```

<!-- ### Set necessary directories

Descriptions for each directory are located below steps that require their usage.

```bash
mkdir output
PTDIR="output/models/vat_pretrain"
GENDIR="output/cb"
RAWDIR="output/cb_raw"
TDIR="output/models/vat_classify"
EDIR="output/models/vat_eval"
``` -->

### Raw Data Parsing & Data Transformations

Training data is drawn from the entire [small dataset](data/data_small.json), and
testing data is drawn from the [2000 pre-selected disjoint sentences](data/disjoint_2000.pkl). In the future, a full corpus from a recent series of presidential debates will be added to this collection to data.

The first time [`pretrain.py`](pretrain.py) is run, code to process raw data will be run if `--refresh_data=True` **or** the code cannot find the stored, processed `.pkl` files containing processed data. Please see the next section for code on running the pre-train file.

### Classifier Fine-Tuning

Once data is processed and dumped into `.pkl` files, [`pretrain.py`](pretrain.py) will continue to build a graph initialized from a pre-trained BERT model. For all of the remaining pre- and adv-training steps, please see [`flags.py`](flags.py) for more information on flag listings and descriptions.

```bash
python3 pretrain.py \
    --cb_output_dir=$PTDIR \
    --bert_model_size=large_wwm \
    --gpu=0
```

`$PTDIR` indicates the location where the pre-trained model should be stored.

### Adversarial Training

```bash
python3 advtrain.py \
    --cb_input_dir=$PTDIR \
    --cb_output_dir=$ADVDIR \
    --gpu=0 \
    --perturb_id=0
```

`$ADVDIR` indicates the location where the adv-trained model should be stored. `perturb_id` can be in the range `[0,7]` and determines which combination of embeddings will be perturbed. Please see [`flags.py`](flags.py) for more information.

### Pretrain CB Language Model

```bash
python3 -u pretrain.py \
    --train_dir=$PTDIR \
    --cb_data_dir=$GENDIR \
    --embedding_dims=300 \
    --rnn_cell_size=1024 \
    --num_candidate_samples=64 \
    --batch_size=64 \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.9999 \
    --max_steps=20000 \
    --max_grad_norm=1.0 \
    --num_timesteps=400 \
    --keep_prob_emb=0.5 \
    --num_classes=3 \
    --normalize_embeddings \
    --bidir_lstm=True \
    --w2v_loc=data/word2vec/GoogleNews-vectors-negative300.bin \
    --transfer_learn_w2v=True \
    0 # which GPU to use [0, 1, ..., n]
```

`$PTDIR` contains the pretrained LSTM language model.

### Train classifier

Most flags stay the same, save for the removal of candidate sampling and the
addition of `pretrained_model_dir`, from which the classifier will load the
pretrained embedding and LSTM variables, and flags related to adversarial
training and classification.

```bash
python3 -u train_classifier.py \
    --train_dir=$TDIR \
    --pretrained_model_dir=$PTDIR \
    --cb_data_dir=$GENDIR \
    --embedding_dims=300 \
    --rnn_cell_size=1024 \
    --cl_num_layers=1 \
    --cl_hidden_size=30 \
    --batch_size=64 \
    --num_classes=3 \
    --learning_rate=0.0005 \
    --learning_rate_decay_factor=0.9998 \
    --max_steps=1650 \
    --max_grad_norm=1.0 \
    --num_timesteps=400 \
    --keep_prob_emb=0.5 \
    --normalize_embeddings \
    --adv_training_method=vat \
    --perturb_norm_length=5.0 \
    --bidir_lstm=True \
    --w2v_loc=data/word2vec/GoogleNews-vectors-negative300.bin \
    --transfer_learn_w2v=True \
    0 # which GPU to use [0, 1, ..., n]
```

`$TDIR` contains the adversarially trained LSTM language model.

### Evaluate on test data

```bash
python3 -u evaluate.py \
    --eval_dir=$EDIR \
    --checkpoint_dir=$TDIR \
    --eval_data=test \
    --run_once \
    --cb_data_dir=$GENDIR \
    --cb_input_dir=$RAWDIR \
    --num_classes=3 \
    --embedding_dims=300 \
    --rnn_cell_size=1024 \
    --batch_size=800 \
    --num_timesteps=400 \
    --normalize_embeddings \
    --bidir_lstm=True \
    --multiclass_metrics=True \
    --w2v_loc=data/word2vec/GoogleNews-vectors-negative300.bin \
    --transfer_learn_w2v=True \
    0 # which GPU to use [0, 1, ..., n]
```

`$EDIR` contains evaluation logs.

## Contributors

Code in this repository was contributed to by:
* Kevin Meng, @kmeng01
* And others