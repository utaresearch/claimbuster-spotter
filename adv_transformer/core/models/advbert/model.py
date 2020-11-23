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
import tensorflow as tf
from bert.model import BertModelLayer
from bert.transformer import TransformerEncoderLayer
from adv_transformer.core.models.advbert.pooler import PoolerLayer
from adv_transformer.core.models.advbert.embeddings import AdvBertEmbeddingsLayer


class AdvBertModelLayer(BertModelLayer):
    class Params(AdvBertEmbeddingsLayer.Params, TransformerEncoderLayer.Params):
        pass

    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, token_type_ids_shape = input_shape
            self.input_spec = [tf.keras.layers.InputSpec(shape=input_ids_shape),
                               tf.keras.layers.InputSpec(shape=token_type_ids_shape)]
        else:
            input_ids_shape = input_shape
            self.input_spec = tf.keras.layers.InputSpec(shape=input_ids_shape)

        self.embeddings_layer = AdvBertEmbeddingsLayer.from_params(
            self.params,
            name="embeddings"
        )

        # create all transformer encoder sub-layers
        self.encoders_layer = TransformerEncoderLayer.from_params(
            self.params,
            name="encoder"
        )

        self.dropout_layer = tf.keras.layers.Dropout(rate=self.params.hidden_dropout)
        self.pooler_layer = PoolerLayer(self.params.hidden_size, name="pooler")

        super(BertModelLayer, self).build(input_shape)

    def call(self, inputs, perturb=None, get_embedding=-1, mask=None, training=None):
        if mask is None:
            mask = self.embeddings_layer.compute_mask(inputs)

        out = self.embeddings_layer(inputs, mask=mask, training=training, get_embedding=get_embedding, perturb=perturb)
        embedding_output, ret_embed = out[0], out[1]

        output = self.encoders_layer(embedding_output, mask=mask, training=training)
        output = self.dropout_layer(output, training=training)

        pooled_output = self.pooler_layer(tf.squeeze(output[:, 0:1, :], axis=1), training=training)

        if get_embedding == -1:
            return pooled_output
        else:
            return ret_embed, pooled_output
