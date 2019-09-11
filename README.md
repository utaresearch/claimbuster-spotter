# Adversarial Claim Spotting
In this repository, we apply adversarial training as as regularization technique for the purpose of determining whether a given claim is a) factual and b) worthy of fact checking.

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

Each training session is predicated upon a pre-trained BERT model as an initialization point. After loading these weights, there are two possible training algorithms:

* Classifier Fine-Tuning ([`pretrain.py`](pretrain.py)): uses vanilla stochastic gradient descent to minimize softmax classification objective into NFS/UFS/CFS class division.
* Adversarial Classifier Fine-Tuning ([`advtrain.py`](advtrain.py)): applies adversarial perturbations to embeddings designated by `--perturb_id` flag (see [`flags.py`](flags.py) for additional details)

Depending on the VRAM capacity of the selected GPU, as well as the predefined batch size, training time can range between 1 and 10 hours. On UTA servers, it may take approximately 6-7 hours to train a `BERT-Large` model.

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

Note that the entire [small dataset](data/data_small.json) will be used for training, and the [disjoint 2000 dataset](data/disjoint_2000.json) will be used for validation.

```bash
python3 pretrain.py \
    --cb_output_dir=$REGDIR \
    --bert_model_size=large_wwm \
    --gpu=0
```

To continue training from another checkpoint, specify the location of the desired model using `$PTDIR`. `$REGDIR` indicates the location of regular training's output.

```bash
python3 pretrain.py \
    --cb_input_dir=$PTDIR \
    --cb_output_dir=$REGDIR \
    --bert_model_size=large_wwm \
    --restore_and_continue=True
    --gpu=0
```

### Adversarial Training

If `$REGDIR` is an empty string, the code will initialize weights from pre-trained BERT. `$ADVDIR` indicates the location where the adv-trained model should be stored. `perturb_id` can be in the range `[0,7]` and determines which combination of embeddings will be perturbed. Please see [`flags.py`](flags.py) for more information.

```bash
python3 advtrain.py \
    --cb_input_dir=$REGDIR \
    --cb_output_dir=$ADVDIR \
    --gpu=0 \
    --perturb_id=0
```

### Performance Evaluation on Test Datasets

Currently, there is only one test dataset available: [disjoint 2000](data/disjoint_2000.json), which was used to evaluate the work of D. Jimenez et al. As previously mentioned, the researchers will be adding a complete presidential debate series to this list. Compatibility updates will follow soon.

```bash
python3 eval.py \
    --cb_output_dir= $EVALDIR \
    --gpu=0
```

`$EVALDIR` can be set to either `$PTDIR` or `$ADVDIR` for evalution of either the pre-trained or adv-trained model, respectively.

### Demonstration on Custom-Input Sentences

**Warning: This file is deprecated! Please do not run until it is fixed in a future push and this warning message is removed!**

Information to come.

## Contributors

Code in this repository was contributed to by:
* Kevin Meng, [`@kmeng01`](https://github.com/kmeng01)
* And others