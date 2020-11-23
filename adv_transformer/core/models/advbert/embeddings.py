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
import params_flow as pf
from bert.embeddings import BertEmbeddingsLayer, EmbeddingsProjector, PositionEmbeddingLayer
from absl import logging


class AdvBertEmbeddingsLayer(BertEmbeddingsLayer):
    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, token_type_ids_shape = input_shape
            self.input_spec = [tf.keras.layers.InputSpec(shape=input_ids_shape),
                               tf.keras.layers.InputSpec(shape=token_type_ids_shape)]
        else:
            input_ids_shape = input_shape
            self.input_spec = tf.keras.layers.InputSpec(shape=input_ids_shape)

        # use either hidden_size for BERT or embedding_size for ALBERT
        embedding_size = self.params.hidden_size if self.params.embedding_size is None else self.params.embedding_size

        self.word_embeddings_layer = tf.keras.layers.Embedding(
            input_dim=self.params.vocab_size,
            output_dim=embedding_size,
            mask_zero=self.params.mask_zero,
            name="word_embeddings"
        )
        if self.params.extra_tokens_vocab_size is not None:
            self.extra_word_embeddings_layer = tf.keras.layers.Embedding(
                input_dim=self.params.extra_tokens_vocab_size + 1,  # +1 is for a <pad>/0 vector
                output_dim=embedding_size,
                mask_zero=self.params.mask_zero,
                embeddings_initializer=self.create_initializer(),
                name="extra_word_embeddings"
            )

        # ALBERT word embeddings projection
        if self.params.embedding_size is not None:
            self.word_embeddings_projector_layer = EmbeddingsProjector.from_params(
                self.params, name="word_embeddings_projector")

        position_embedding_size = embedding_size if self.params.project_position_embeddings else self.params.hidden_size

        if self.params.use_token_type:
            self.token_type_embeddings_layer = tf.keras.layers.Embedding(
                input_dim=self.params.token_type_vocab_size,
                output_dim=position_embedding_size,
                mask_zero=False,
                name="token_type_embeddings"
            )
        if self.params.use_position_embeddings:
            self.position_embeddings_layer = PositionEmbeddingLayer.from_params(
                self.params,
                name="position_embeddings",
                hidden_size=position_embedding_size
            )

        self.layer_norm_layer = pf.LayerNormalization(name="LayerNorm")
        self.dropout_layer = tf.keras.layers.Dropout(rate=self.params.hidden_dropout)

        super(BertEmbeddingsLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, get_embedding=-1, perturb=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids = inputs
            token_type_ids = None

        input_ids = tf.cast(input_ids, dtype=tf.int32)

        if self.extra_word_embeddings_layer is not None:
            token_mask = tf.cast(input_ids >= 0, tf.int32)
            extra_mask = tf.cast(input_ids < 0, tf.int32)
            token_ids = token_mask * input_ids
            extra_tokens = extra_mask * (- input_ids)
            token_output = self.word_embeddings_layer(token_ids)
            extra_output = self.extra_word_embeddings_layer(extra_tokens)
            tok_embed = tf.add(token_output,
                               extra_output * tf.expand_dims(tf.cast(extra_mask, tf.keras.backend.floatx()), axis=-1))
            embedding_output = tok_embed
        else:
            tok_embed = self.word_embeddings_layer(input_ids)
            embedding_output = tok_embed

        # ALBERT: for brightmart/albert_zh weights - project only token embeddings
        if not self.params.project_position_embeddings:
            if self.word_embeddings_projector_layer:
                embedding_output = self.word_embeddings_projector_layer(embedding_output)

        assert token_type_ids is not None, 'token_type_ids cannot be none'
        if token_type_ids is not None:
            token_type_ids = tf.cast(token_type_ids, dtype=tf.int32)
            seg_embed = self.token_type_embeddings_layer(token_type_ids)
            embedding_output += seg_embed

        assert self.position_embeddings_layer is not None
        if self.position_embeddings_layer is not None:
            seq_len = input_ids.shape.as_list()[1]
            emb_size = embedding_output.shape[-1]

            pos_embeddings = self.position_embeddings_layer(seq_len)
            # broadcast over all dimension except the last two [..., seq_len, width]
            broadcast_shape = [1] * (embedding_output.shape.ndims - 2) + [seq_len, emb_size]
            pos_embed = tf.reshape(pos_embeddings, broadcast_shape)

            embedding_output += pos_embed

        ret_embed = tf.constant(0)

        if get_embedding != -1:
            all_embeddings = {
                'tok': tok_embed,
                'seg': seg_embed,
                'pos': pos_embed
            }
            perturbable = [('pos', 'seg', 'tok'), ('pos', 'seg'), ('pos', 'tok'), ('seg', 'tok'), ('pos',), ('seg',),
                           ('tok',)]
            cfg = perturbable[get_embedding]
            logging.info(cfg)

            changed = False
            for el in cfg:
                if not changed:
                    ret_embed = all_embeddings[el]
                    changed = True
                else:
                    ret_embed += all_embeddings[el]

            embedding_output = ret_embed
            logging.info(embedding_output)

            diff = set(perturbable[0]).difference(cfg)
            for el in diff:
                embedding_output += all_embeddings[el]

        else:
            embedding_output = tok_embed + seg_embed + pos_embed

        if perturb is not None:
            embedding_output += perturb

        embedding_output = self.layer_norm_layer(embedding_output)
        embedding_output = self.dropout_layer(embedding_output, training=training)

        # ALBERT: for google-research/albert weights - project all embeddings
        if self.params.project_position_embeddings:
            if self.word_embeddings_projector_layer:
                embedding_output = self.word_embeddings_projector_layer(embedding_output)

        return [embedding_output, ret_embed]  # [B, seq_len, hidden_size]
