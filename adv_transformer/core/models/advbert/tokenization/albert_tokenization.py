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
from bert.tokenization.albert_tokenization import FullTokenizer, WordpieceTokenizer, encode_pieces, printable_text, convert_by_vocab


def encode_ids(sp_model, text, sample=False):
    pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
    pieces = ['[CLS]'] + pieces + ['[SEP]']
    ids = [sp_model.PieceToId(piece) for piece in pieces]

    return ids


class AdvFullTokenizer(FullTokenizer):
    def __init__(self, vocab_file, do_lower_case=True, spm_model_file=None):
        super().__init__(vocab_file, do_lower_case, spm_model_file)

    def convert_tokens_to_ids(self, tokens):
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        if self.sp_model:
            return [self.sp_model.PieceToId(
                printable_text(token)) for token in tokens]
        else:
            return convert_by_vocab(self.vocab, tokens)


class AdvWordpieceTokenizer(WordpieceTokenizer):
    def __init__(self, vocab, unk_token="<unk>", max_input_chars_per_word=200):
        super().__init__(vocab, unk_token, max_input_chars_per_word)
