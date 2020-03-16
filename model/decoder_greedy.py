"""
This module holds our greedy search decoder
"""
import sys

sys.path.append('..')
import torch
from torch import nn
from vocab import SOS_TOKEN

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class GreedySearchDecoder(nn.Module):
    """Greedy searcher"""

    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=DEVICE,
                                   dtype=torch.long) * SOS_TOKEN
        all_tokens = torch.zeros([0], device=DEVICE, dtype=torch.long)
        all_scores = torch.zeros([0], device=DEVICE)
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores
