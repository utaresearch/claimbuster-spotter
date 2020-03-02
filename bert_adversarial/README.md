# Adversarial Claim Spotting
We present our application of adversarial training as a regularization technique, on top of a transformer network, for the purpose of determining whether a given claim is factual and worthy of fact-checking.

## Table of Contents
1. [Requirements](#requirements)
2. [Experimental Setup](#experimental-setup)
3. [Code Overview](#code-overview)
4. [API Wrapper](#api-wrapper)
5. [Contributions](#contributors)

## Requirements

Please see [`requirements.txt`](bert-adversarial/requirements.txt) for a list of required Python packages. You may use `pip3 install -r requirements.txt` to install them at once.

## Experimental Setup

* Intel(R) Xeon(R) CPU E5-2695 v4 @ 2.10GHz
* 256GB RAM
* 4x Nvidia GTX 1080Ti (12GB RAM each)

## Code Overview

This section provides a high-level overview of this repository, as well as details regarding the functions of each source file.

### Pre-processing

Preprocessing is accomplished by [`train.py`](bert-adversarial/train.py). Sentences are extracted from `./data/`, transformed based on various flags defined in [`flags.py`](flags.py), and converted into the token/segment/mask format required by BERT.

### Training

Each training session is predicated upon a pre-trained BERT model as an initialization point. After loading these weights, we offer two possible training algorithms:

* Classifier Fine-Tuning (`--cs_adv_train=False`): uses vanilla stochastic gradient descent to minimize softmax classification objective into NFS/UFS/CFS class division.
* Adversarial Classifier Fine-Tuning (`--cs_adv_train=True`): applies adversarial perturbations to embeddings and uses stochastic gradient descent to minimize error between each resulting prediction and its corresponding ground-truth.

Depending on the VRAM capacity of the selected GPU, as well as the predefined batch size and number of frozen layers, training time can range between 1 and 10 hours. On an Nvidia GeForce GTX 1080Ti, it takes approximately 3 minutes/epoch to train a `BERT-Base` model using regular optimization. Adversarial training doubles the time required.

### Evaluations

We perform 4-fold cross validation on our dataset. At the end of all 4 folds we output a classification report of the aggregated results across all 4 folds.

If you wish to evaluate a model after training it on a different dataset, then you can call `eval.py`, but you'll need to specify the following flags:

| Flag | Value | Special Notes |
| :------------- | :------------- | :------------- |
| `cs_reg_train_file`  | `../data/two_class/cfs_ncs_ratio_study_train_25.json`  | Arbitrary value since flag needs to be specified. Does not affect evaluation results.|
| `cs_reg_test_file` | `path/to/your/test_data.json`  | |
| `cs_k_fold` | 1 | Used to specify that data should be loaded from the locations specified above. Otherwise, the k-fold evaluation data will be loaded.|
| `cs_refresh_data` | True | Used to specify that the data must be refreshed, and ensure no cached version is used. |
| `cs_gpu` | `int: GPU ID of GPU to use` | Optional flag, the default value is `0`. |

### Interactive ClaimSpotter Demo

Using [`demo.py`](bert-adversarial/demo.py), users can input individual sentences into the command line, and the model will produce an inference result on the given sentence. Each sentence should take under 100ms to process.

```bash
# From the root folder execute:
python3 -m bert_adversarial.demo.py \
    --cs_model_dir=$MDIR \
    --cs_gpu=0
```

### Command-Line Flags

All flags are defined and editable in [`core/utils/flags.py`](bert-adversarial/core/utils/flags.py).

## End-to-End Claim Spotting Procedure

### Clone GitHub repository
```bash
git clone git@github.com:idirlab/claimspotter.git
```

### CD to current directory

From this point forwards, all directories are referenced relative to the project
root.

```bash
cd adversarial-claimspotting
```

### Fetch dependencies

Because BERT pre-trained files are inconvenient/impossible to track with Git, they must be downloaded at time of use. There is a convenient pre-written script for this purpose.

```bash
chmod +x ./data/get_bert.sh
chmod +x ./data/get_albert.sh
chmod +x ./dependencies.sh
./dependencies.sh
```

### Raw Data Parsing & Data Transformations

Training data is drawn from the entire [small dataset](data/data_small.json), and
testing data is drawn from the [2000 pre-selected disjoint sentences](data/disjoint_2000.pkl). In the future, a full corpus from a recent series of presidential debates will be added to this collection to data.

When [`train.py`](bert-adversarial/train.py) is run, code to process raw data will be run if `--cs_refresh_data=True` **or** the code cannot find the stored, processed `.pkl` files containing processed data. Please see the next section for code on running the pre-train file.

### Classifier Fine-Tuning

Once data is processed and dumped into `.pkl` files, [`train.py`](bert-adversarial/train.py) will continue to build a graph initialized from a pre-trained BERT model. For all of the remaining pre- and adv-training steps, please see [`flags.py`](flags.py) for more information on flag listings and descriptions.

**`$MDIR`** indicates the location where the trained model should be stored. 

```bash
# From the root folder execute:
python3 -m bert_adversarial.train \
    --cs_model_dir=$MDIR \
    --cs_adv_train=False \
    --cs_gpu=0
```

### Adversarial Training

As with regular training, `$MDIR` indicates the location where the trained model should be stored. `perturb_id` can be in the range `[0, 6]` and determines which combination of embeddings will be perturbed. Please see [`flags.py`](flags.py) for more information.

```bash
# From the root folder execute:
python3 -m bert_adversarial.train \
    --cs_model_dir=$MDIR \
    --cs_adv_train=True \
    --cs_gpu=0
```

### Restore and Continue Training

This option is not available for k-fold evaluation training sessions. To continue training from a previous checkpoint, specify that `--restore_and_continue=True`. This will retrieve weights stored in `$MDIR` and continue training in the same folder. Epoch numbers are continuous between training sessions. If the flag is false (as it is by default), the code will initialize weights from a pre-trained BERT model.

Continued training does not depend on the algorithm used to train the preceding model. In other words, one may continue adversarially training a previously regularly trained model, and vice-versa. However, transformer sizes *must* be consistent when restoring and continuing.

Below is an example of using `restore_and_continue` on adversarial training.

```bash
# From the root folder execute:
python3 -m bert_adversarial.train \
    --cs_model_dir=$MDIR \
    --cs_adv_train=True
    --cs_gpu=0 \
    --cs_restore_and_continue=True
```

## API Wrapper

We provide an API wrapper in [core/api/api_wrapper.py](bert-adversarial/core/api/api_wrapper.py) to enable easy integration into other applications. There are two simple query functions that extract inference information for a single sentence. Below is a sample usage scenario:

```python
from bert_adversarial.core.api.api_wrapper import ClaimSpotterAPI

api = ClaimSpotterAPI()
sentence = "ClaimBuster is a state-of-the-art, end-to-end fact-checking system."
api_result = api.single_sentence_query(sentence)  # Returns array w/ class probabilities

api_result_2 = api.subscribe_cmdline_query()  # Collects/processes cmdline input
```

## Sample Web API

In `app.py` you'll find code to host a trained model as a web application. The scoring function is accessible via endpoints which provide users with different options to score different text sources.

```bash
# From the root folder execute:
python3 -m bert_adversarial.app \
    --cs_model_dir=$MDIR \
    --cs_gpu=0
```
