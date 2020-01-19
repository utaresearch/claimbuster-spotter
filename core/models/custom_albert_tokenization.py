from .bert2.tokenization.albert_tokenization import encode_ids, preprocess_text
from ..utils.flags import FLAGS
import sentencepiece as spm
import os


class CustomAlbertTokenizer:
    def __init__(self):
        self.model = spm.SentencePieceProcessor()
        self.model.load(os.path.join(FLAGS.cs_model_loc, "30k-clean.model"))

    def tokenize_array(self, inp):
        return [encode_ids(self.model, preprocess_text(x, lower=True)) for x in inp]
