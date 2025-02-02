import torch
import torch.nn as nn
from torch import Tensor

import modules
from modules import PredictionLayer


class Transformer(nn.Module):
    def __init__(self,
                 input_vocab_size: int,
                 output_vocab_size: int,
                 input_sequence_length: int,
                 output_sequence_length: int,
                 dimension: int,
                 encoder_iterations: int,
                 decoder_iterations: int,
                 encoder_heads: int,
                 decoder_heads: int,
                 dropout: float,
                 feedforward_dimension: int
                ) -> None:
        super().__init__()
        self.dimension = dimension
        self.encoder_iterations = encoder_iterations
        self.decoder_iterations = decoder_iterations
        self.dropout = dropout

        self.encoder_input_embedding = modules.Embedding(
            input_vocab_size,
            dimension
        )
        self.encoder_positional_embedding = modules.Embedding(
            input_sequence_length,
            dimension
        )
        self.decoder_input_embedding = modules.Embedding(
            output_vocab_size,
            dimension
        )
        self.decoder_positional_embedding = modules.Embedding(
            output_sequence_length,
            dimension
        )

        self.encoder_attention_layers = nn.ModuleList([
            modules.MaskedMultiHeadAttention(dimension, encoder_heads)
            for _ in range(encoder_iterations)
        ])
        self.encoder_feed_forward_networks = nn.ModuleList([
            modules.FeedForward(dimension, feedforward_dimension)
            for _ in range(encoder_iterations)
        ])
        self.decoder_self_attention_layers = nn.ModuleList([
            modules.MaskedMultiHeadAttention(dimension, decoder_heads)
            for _ in range(decoder_iterations)
        ])
        self.decoder_cross_attention_layers = nn.ModuleList([
            modules.MaskedMultiHeadAttention(dimension, decoder_heads)
            for _ in range(decoder_iterations)
        ])
        self.decoder_feed_forward_networks = nn.ModuleList([
            modules.FeedForward(dimension, feedforward_dimension)
            for _ in range(decoder_iterations)
        ])

        self.encoder_attention_layer_norms = nn.ModuleList([
            modules.LayerNorm(dimension) for _ in range(encoder_iterations)
        ])
        self.encoder_feed_forward_network_layer_norms = nn.ModuleList([
            modules.LayerNorm(dimension) for _ in range(encoder_iterations)
        ])
        self.decoder_self_attention_layer_norms = nn.ModuleList([
            modules.LayerNorm(dimension) for _ in range(decoder_iterations)
        ])
        self.decoder_cross_attention_layer_norms = nn.ModuleList([
            modules.LayerNorm(dimension) for _ in range(decoder_iterations)
        ])
        self.decoder_feed_forward_network_layer_norms = nn.ModuleList([
            modules.LayerNorm(dimension) for _ in range(decoder_iterations)
        ])

        self.prediction_layer = PredictionLayer(dimension, output_vocab_size)



    # sequence shape: (B, S) dtype = int (from Tokenizer)
    def encode(self,sequence, mask: Tensor) -> Tensor:
        batch_size, sequence_length = sequence.shape

        # (B, S) -> (B, S, D)
        sequence = self.encoder_input_embedding(sequence)
        # pos.shape: (S,)
        positions = torch.arange(sequence_length)
        # (S,) -> (B,S)
        # unsqueeze  increases dimension to (1,S), expand then duplicates the -1 dim (S) B times
        positions = positions.unsqueeze(0).expand(batch_size,-1)

        # (B,S,D) + emb(B,S) = (B,S,D)
        sequence += self.encoder_positional_embedding(positions)
        sequence = modules.dropout(sequence, self.dropout)


        for i in range(self.encoder_iterations):
            # (B,S) -> (B,S,D)
            sequence = self.encoder_attention_layer_norms[i](
                sequence + modules.dropout(self.encoder_attention_layers[i](
                    {"query_": sequence, "keys_": sequence, "values_": sequence},
                    mask
                ), self.dropout)
            )
            sequence = self.encoder_feed_forward_network_layer_norms[i](
                sequence + modules.dropout(
                    self.encoder_feed_forward_networks[i](sequence),
                    self.dropout
                )
            )
        return sequence

    def decode(self,
               sequence,
               encoder_output,
               decoder_mask: Tensor = None,
               encoder_mask: Tensor = None
               ):
        batch_size, sequence_length = sequence.shape

        # (batch_size, seq_length) -> (batch_size, seq_length, dimension)
        sequence = self.decoder_input_embedding(sequence)
        positions = torch.arange(sequence_length)
        positions = positions.unsqueeze(0).expand(batch_size,-1)
        sequence += self.decoder_positional_embedding(positions)


        for i in range(self.decoder_iterations):
            sequence = self.decoder_self_attention_layer_norms[i](
                sequence + modules.dropout(
                    self.decoder_self_attention_layers[i](
                        {
                            "query_": sequence,
                            "keys_": sequence,
                            "values_": sequence
                        },
                        decoder_mask
                    )
                )
            )
            sequence = self.decoder_cross_attention_layer_norms[i](
                sequence + modules.dropout(
                    self.decoder_cross_attention_layers[i](
                        {
                            "query_": sequence,
                            "keys_": encoder_output,
                            "values_": encoder_output
                        },
                        encoder_mask
                    )
                )
            )
            sequence = self.decoder_feed_forward_network_layer_norms[i](
                sequence + modules.dropout(
                    self.decoder_feed_forward_networks[i](sequence),
                    self.dropout
                )
            )
            return sequence

    def predict(self, sequence):
        return self.prediction_layer(sequence)
