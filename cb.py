"""
Main chatbot entrypoint.
"""
import os
import sys

sys.path.insert(1, 'model')
### PYTORCH
import torch
from torch import nn, optim
### LOAD CORPUS
import corpus_composer
import data_reader
from util import config, normalize_string

import model

from encoder_rnn import EncoderRNN
from decoder_rnn import LuongAttnDecoderRNN
from decoder_greedy import GreedySearchDecoder

corpus_name = config()['DEFAULT']['corpus']
if not os.path.exists(os.path.join(
        os.path.dirname(__file__),
        config()['data']['data_path'], corpus_name + '.train')):
    corpus_composer.build_set(corpus_name)
voc, pairs = data_reader.load_prepare_data(corpus_name,
                                           f'data/{corpus_name}.train')

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

print(f"Using device {DEVICE}")

### SET UP MODEL
model_name = 'cb_model'
attn_model = 'dot'

HIDDEN_SIZE = int(config()['DEFAULT']['hidden_size'])

encoder_n_layers = int(config()['DEFAULT']['encoder_n_layers'])
decoder_n_layers = int(config()['DEFAULT']['decoder_n_layers'])

dropout = float(config()['DEFAULT']['dropout'])
batch_size = int(config()['DEFAULT']['batch_size'])

# Set checkpoint to load from; set to None if starting from scratch
# load_file_name = None
checkpoint_iter = 100000
save_dir = config()['DEFAULT']['save_dir']
load_file_name = \
    os.path.join(os.path.dirname(__file__), save_dir, model_name, corpus_name,
                 f'{encoder_n_layers}-{decoder_n_layers}_{HIDDEN_SIZE}',
                 f'{checkpoint_iter}_checkpoint.ptmodel')

if load_file_name:
    checkpoint = torch.load(load_file_name)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
embedding = nn.Embedding(voc.num_words, HIDDEN_SIZE)
if load_file_name:
    embedding.load_state_dict(embedding_sd)

encoder = EncoderRNN(HIDDEN_SIZE, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, HIDDEN_SIZE, voc.num_words,
                              decoder_n_layers, dropout)
if load_file_name:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

encoder = encoder.to(DEVICE)
decoder = decoder.to(DEVICE)

### TRAIN MODEL
clip = 50.0
teacher_forcing_ratio = float(config()['DEFAULT']['teacher_forcing_ratio'])
learning_rate = float(config()['DEFAULT']['learning_rate'])
decoder_learning_ratio = 5.0
n_iteration = int(config()['DEFAULT']['number_iters'])
# print_every = 1
print_every = int(config()['DEFAULT']['PRINT_EVERY'])
save_every = int(config()['DEFAULT']['SAVE_EVERY'])

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if load_file_name:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

if USE_CUDA:
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

searcher = GreedySearchDecoder(encoder, decoder)

def eval_sentence(sentence: str) -> str:
    output_words = model.evaluate(encoder, decoder, searcher, voc,
                                  normalize_string(sentence))
    output_words[:] = [x for x in output_words if not (x == 'EOS' or
                                                       x == 'PAD')]
    return ' '.join(output_words)


if __name__ == "__main__":
    # Run training iterations
    print('Training iters ...')
    model.train_iters(
        model_name, voc, pairs, encoder, decoder, encoder_optimizer,
        decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir,
        n_iteration, batch_size, print_every, save_every, clip, corpus_name,
        load_file_name
    )

    searcher = GreedySearchDecoder(encoder, decoder)
    model.evaluate_input(encoder, decoder, searcher, voc)
