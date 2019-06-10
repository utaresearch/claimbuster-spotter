# Adversarial Claim Spotting
This is an application of techniques described in [*Adversarial Training Methods for Semi-Supervised Text Classification*](https://arxiv.org/abs/1605.07725) for the purpose of determining whether a given claim is a) factual and b) worthy of fact checking.

**WARNING! This documentation is deprecated. Updates are coming soon.**

## Requirements

* Python >= 3.0
* imbalanced_learn==0.4.3
* nltk==3.3
* pycontractions==1.0.1
* tqdm==4.11.2
* gensim==3.4.0
* spacy==2.0.11
* tensorflow==1.13.1
* numpy==1.16.4
* imblearn==0.0
* scikit_learn==0.21.2

## Experimental Setup

* Intel(R) Xeon(R) CPU E5-2695 v4 @ 2.10GHz
* 256GB RAM
* 4x Nvidia GTX 1080Ti
* 4TB HDD

## Code Overview

Data pre-processing is carried out prior to training the models.
Then, each source file builds a `VatxtModel`, defined in `graphs.py`, which in turn uses graph building blocks
defined in `inputs.py` (defines input data reading and parsing), `layers.py`
(defines core model components), and `adversarial_losses.py` (defines
adversarial training losses). The training loop itself is defined in
`train_utils.py`. Finally, `evaluation.py` outputs data regarding model performance.

### Pre-processing

* Data transformations and reformatting: [`preprocess_data.py`](preprocess_data.py)
* Vocabulary generation: [`gen_vocab.py`](gen_vocab.py)
* Data generation: [`gen_data.py`](gen_data.py)

### Training

* Pretraining: [`pretrain.py`](train.py)
* Classifier Training: [`train_classifier.py`](train_classifier.py)

### Evaluations

* Performance Evaluation: [`evaluate.py`](evaluate.py)

### Command-Line Flags

* Flags related to distributed training and the training loop itself are defined
in [`train_utils.py`](train_utils.py).
* Flags related to model hyperparameters are defined in [`graphs.py`](graphs.py).
* Flags related to adversarial training are defined in [`adversarial_losses.py`](adversarial_losses.py).
* Flags particular to each job are defined in the main source files.
* Command-line flags defined in [`document_generators.py`](data/document_generators.py) control the manner with which documents are generated.

## Different Modes for Executing Code

In [`utils/`](utils/), there are 3 files with varying purposes, as listed below. [`run_eval.sh`](utils/run_eval.sh) is used for official evaluation.
* [`run_sm.sh`](utils/run_sm.sh): Uses [small dataset](data/data_small.json) split into training/testing
* [`run_lg.sh`](utils/run_lg.sh): Uses [large dataset](data/data_large.json) split into training/testing
* [`run_eval.sh`](utils/run_eval.sh): Uses the entire [small dataset](data/data_small.json) for training and the [2000 pre-selected disjoint sentences](data/disjoint_2000.pkl) for evaluation. These are the commands described below in the example procedure.

## End-to-End Claim Spotting Procedure

### Clone GitHub repository
```bash
git clone https://github.com/idirlab/GANclaimspotting.git
```

### Install dependencies
```bash
pip3 install -r requirements.txt
```

### CD to current directory

From this point forwards, all directories are referenced relative to the project
root.

```bash
cd GANclaimspotting
```

### Fetch word2vec and spaCy models

Since the word2vec and spaCy binaries are too large to track even with Git LFS, they must be downloaded at time of use.

```bash
python3 -m spacy download en_core_web_lg
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz \
    -O data/word2vec/GoogleNews-vectors-negative300.bin.gz
gunzip data/word2vec/GoogleNews-vectors-negative300.bin.gz
```

### Set necessary directories

Descriptions for each directory are located below steps that require their usage.

```bash
mkdir output
PTDIR="output/models/vat_pretrain"
GENDIR="output/cb"
RAWDIR="output/cb_raw"
TDIR="output/models/vat_classify"
EDIR="output/models/vat_eval"
```

### Parse data from JSON file and apply data transformations

Training data is drawn from the entire [small dataset](data/data_small.json), and
testing data is drawn from the [2000 pre-selected disjoint sentences](data/disjoint_2000.pkl).

```bash
python3 -u preprocess_eval_data.py \
    --output_dir=$RAWDIR \
    --train_loc=data/data_small.json \
    --diff_test_loc=True \
    --test_loc=data/disjoint_2000.pkl \
    --w2v_loc=data/word2vec/GoogleNews-vectors-negative300.bin \
    --ner_loc1=data/ner/classifiers/english.muc.7class.distsim.crf.ser.gz \
    --ner_loc2=data/ner/stanford-ner.jar \
    --noun_rep=False \
    --full_tags=False \
    --ner_stanford=False \
    --ner_spacy=True \
    --num_classes=3
```

`$RAWDIR` contains transformed data in a hierarchical structure.

### Generate vocabulary

```bash
python3 -u gen_vocab.py \
    --output_dir=$GENDIR \
    --dataset=cb \
    --cb_input_dir=$RAWDIR \
    --lowercase=False \
    --large_dataset=False \
    --include_validation=True \
    --validation_pct=5 \
    --num_classes=3
```

`$GENDIR` contains vocabulary and frequency files used to normalize adversarial perturbations.

###  Generate training, validation, and test data

```bash
python3 -u gen_data.py \
    --output_dir=$GENDIR \
    --dataset=cb \
    --cb_input_dir=$RAWDIR \
    --lowercase=False \
    --label_gain=False \
    --large_dataset=False \
    --validation_pct=5 \
    --num_classes=3
```

`$GENDIR` contains TFRecords files for training and evaluation.

### Pretrain CB Language Model

```bash
python3 -u pretrain.py \
    --train_dir=$PTDIR \
    --data_dir=$GENDIR \
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
    --data_dir=$GENDIR \
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
    --data_dir=$GENDIR \
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

* Original Implementation in Research Paper
    * Ryan Sepassi, @rsepassi
    * Andrew M. Dai, @a-dai
    * Takeru Miyato, @takerum
* Application to Detection of Salient Factual Statements
    * Kevin Meng, @kmeng01
