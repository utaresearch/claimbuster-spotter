# Copyright (C) 2020 IDIR Lab - UT Arlington
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License v3 as published by
#     the Free Software Foundation.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact Information:
#     See: https://idir.uta.edu/cli.html
#
#     Chengkai Li
#     Box 19015
#     Arlington, TX 76019
#
import sentencepiece as spm
import os
from bert.tokenization.albert_tokenization import preprocess_text
from adv_transformer.core.models.advbert.tokenization.albert_tokenization import encode_ids
from adv_transformer.core.utils.flags import FLAGS


class CustomAlbertTokenizer:
    def __init__(self):
        self.model = spm.SentencePieceProcessor()
        self.model.load(os.path.join(FLAGS.cs_model_loc, "30k-clean.model"))

    def tokenize_array(self, inp):
        return [encode_ids(self.model, preprocess_text(x, lower=True)) for x in inp]
