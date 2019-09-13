# Adversarial Claim Spotting
In this repository, we apply adversarial training as as regularization technique for the purpose of determining whether a given claim is a) factual and b) worthy of fact checking.

## Table of Contents
1. [Requirements](#requirements)
2. [Experimental Setup](#experimental-setup)
3. [Code Overview](#code-overview)
4. [API Wrapper](#api-wrapper)
5. [Contributions](#contributors)

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

Depending on the VRAM capacity of the selected GPU, as well as the predefined batch size, training time can range between 1 and 10 hours. On UTA servers, it may take approximately 3 hours to train a `BERT-Base` model and 6-7 hours to train a `BERT-Large` model using regular optimization. Adversarial training doubles the time required.

### Evaluations

We output F1 scores on the `disjoint_2000.json` dataset using [`eval.py`](eval.py). In the future, a full corpus of quotes from a recent presidential campaign debate series will be used to quantify the in-the-wild reliability of the claim-spotting model.

### Interactive ClaimBuster Demo

Using [`demo.py`](demo.py), users can input individual sentences into the command line, and the model will produce an inference result on the inputted sentence. Each sentence should take under 100ms to process.

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

Because word2vec binaries and BERT pre-trained files are inconvenient/impossible to track with Git, they must be downloaded at time of use. There is a convenient pre-written script for this purpose.

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

When [`pretrain.py`](pretrain.py) is run, code to process raw data will be run if `--refresh_data=True` **or** the code cannot find the stored, processed `.pkl` files containing processed data. Please see the next section for code on running the pre-train file.

### Classifier Fine-Tuning

Once data is processed and dumped into `.pkl` files, [`pretrain.py`](pretrain.py) will continue to build a graph initialized from a pre-trained BERT model. For all of the remaining pre- and adv-training steps, please see [`flags.py`](flags.py) for more information on flag listings and descriptions.

Note that the entire [small dataset](data/data_small.json) will be used for training, and the [disjoint 2000 dataset](data/disjoint_2000.json) will be used for validation.

`$MDIR` indicates the location where the trained model should be stored. 

```bash
python3 pretrain.py \
    --cb_model_dir=$MDIR \
    --bert_model_size=large_wwm \
    --gpu=0
```

### Adversarial Training

As with regular training, `$MDIR` indicates the location where the trained model should be stored. `perturb_id` can be in the range `[0,7]` and determines which combination of embeddings will be perturbed. Please see [`flags.py`](flags.py) for more information.

```bash
python3 advtrain.py \
    --cb_model_dir=$MDIR \
    --bert_model_size=large_wwm \
    --gpu=0 \
    --perturb_id=0
```

### Restore and Continue Training

To continue training from a previous checkpoint, specify that `--restore_and_continue=True`. This will retrieve weights stored in `$MDIR` and continue training in the same folder. Epoch numbers are continuous between training sessions. If the flag is false (as it is by default), the code will initialize weights from a pre-trained BERT model.

Continued training does not depend on the algorithm used to train the preceding model. In other words, one may continue adversarially training a previously regularly trained model, and vice-versa. However, transformer sizes *must* be consistent when restoring and continuing.

Below is an example of using `restore_and_continue` on adversarial training.

```bash
python3 advtrain.py \
    --cb_model_dir=$MDIR \
    --bert_model_size=large_wwm \
    --gpu=0 \
    --perturb_id=0 \
    --restore_and_continue=True
```

### Performance Evaluation on Test Datasets

Currently, there is only one test dataset available: [disjoint 2000](data/disjoint_2000.json), which was used to evaluate the work of D. Jimenez et al. As previously mentioned, the researchers will be adding a complete presidential debate series to this list. Compatibility updates will follow soon.

```bash
python3 eval.py \
    --cb_model_dir=$MDIR \
    --bert_model_size=large_wwm \
    --gpu=0
```

Either the pre-trained or adv-trained model can be evaluated using this code.

### Demonstration on Custom-Input Sentences

Running the follow code will open an interface to input individual sentences for claim scoring.

```bash
python3 demo.py \
    --cb_model_dir=$MDIR \
    --bert_model_size=large_wwm \
    --gpu=0
```

## API Wrapper

We provide an API wrapper in [api_wrapper.py](api_wrapper.py) to enable easy integration into other applications. There are two simple query functions that extract inference information for a single sentence. Below is a sample usage scenario:

```python
from api_wrapper import ClaimBusterAPI

api = ClaimBusterAPI()
sentence = "ClaimBuster is a state-of-the-art, end-to-end fact-checking system."
api_result = api.direct_sentence_query(sentence)  # Returns array w/ class probabilities

api_result_2 = api.subscribe_cmdline_query()  # Collects/processes cmdline input
```

## Contributors

Code in this repository was contributed to by:
* Kevin Meng, [`@kmeng01`](https://github.com/kmeng01)
* And others