import sentencepiece as spm
import sys
import os
sys.path.append('..')
from flags import FLAGS
from models.xlnet.prepro_utils import preprocess_text, encode_ids

# some code omitted here...
# initialize FLAGS

text = "An input text string."

sp_model = spm.SentencePieceProcessor()
sp_model.Load(os.path.join('..', FLAGS.xlnet_model_loc, 'spiece.model'))
text = preprocess_text(text, lower=False)
ids = encode_ids(sp_model, text)

print(text, ids)