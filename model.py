"""
Model entrypoint.
"""
import random
import os

### PYTORCH SETUP/IMPORT
import torch
from torch import nn
### CONFIG FILE
from util import config
### DATA_READER
import data_reader
### VOCAB
import vocab

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

### CONFIG CONSTANTS
MAX_LENGTH = int(config()['DEFAULT']['MAX_LENGTH'])

TEACHER_FORCING_RATIO = float(config()['DEFAULT']['teacher_forcing_ratio'])

HIDDEN_SIZE = int(config()['DEFAULT']['hidden_size'])


def mask_nll_loss(inp, target, mask):
    """
    Masks the negatively logged likelihood of our data
    """
    n_total = mask.sum()
    cross_entropy = -torch.log(
        torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(DEVICE)
    return loss, n_total.item()


def train(input_variable, lengths, target_variable, mask, max_target_len,
          encoder, decoder, embedding, encoder_optimizer, decoder_optimizer,
          batch_size, clip, max_length=MAX_LENGTH):
    """Train step"""
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = input_variable.to(DEVICE)
    lengths = lengths.to(DEVICE)
    target_variable = target_variable.to(DEVICE)
    mask = mask.to(DEVICE)

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    decoder_input = torch.LongTensor(
        [[vocab.SOS_TOKEN for _ in range(batch_size)]])
    decoder_input = decoder_input.to(DEVICE)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = random.random() < TEACHER_FORCING_RATIO

    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            decoder_input = target_variable[t].view(1, -1)

            mask_loss, n_total = mask_nll_loss(decoder_output,
                                               target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor(
                [[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(DEVICE)

            mask_loss, n_total = mask_nll_loss(decoder_output,
                                               target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total

    loss.backward()

    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def train_iters(model_name, voc, pairs, encoder, decoder, encoder_optimizer,
                decoder_optimizer, embedding, encoder_n_layers,
                decoder_n_layers, save_dir, n_iteration, batch_size,
                print_every, save_every, clip, corpus_name, load_file_name):
    """Train the model for n_iteration iterations and other specified
    arguments """
    print('Preparing batches...')
    training_batches = [data_reader.batch_to_train_data(
        voc, [random.choice(pairs) for _ in range(batch_size)])
        for _ in range(n_iteration)]
    print('Initializing...')
    start_iteration = 1
    print_loss = 0
    if load_file_name:
        checkpoint = torch.load(load_file_name)
        start_iteration = checkpoint['iteration'] + 1

    print('Training...')
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]

        input_variable, lengths, target_variable, mask, max_target_len = \
            training_batch

        loss = train(input_variable, lengths, target_variable, mask,
                     max_target_len, encoder, decoder, embedding,
                     encoder_optimizer,
                     decoder_optimizer, batch_size, clip)

        print_loss += loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print(
                "Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}"
                .format(iteration, iteration / n_iteration * 100,
                        print_loss_avg))
            print_loss = 0

        if iteration % save_every == 0:
            directory = \
                os.path.join(
                    save_dir, model_name, corpus_name,
                    f'{encoder_n_layers}-{decoder_n_layers}_{HIDDEN_SIZE}')
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory,
                            '{}_{}.ptmodel'.format(iteration, 'checkpoint')))
    print("Done Training!")


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    """Format input sentence as a batch"""
    indeces_batch = [data_reader.indeces_from_sentence(voc, sentence)]

    lengths = torch.tensor([len(indeces) for indeces in indeces_batch])

    input_batch = torch.LongTensor(indeces_batch).transpose(0, 1)

    input_batch = input_batch.to(DEVICE)
    lengths = lengths.to(DEVICE)

    tokens, _ = searcher(input_batch, lengths, max_length)

    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluate_input(encoder, decoder, searcher, voc):
    """Test talking to the user"""
    while 1:
        try:
            input_sentence = input('> ')
            if input_sentence == 'q' or input_sentence == 'quit':
                break
            output_words = evaluate(encoder, decoder, searcher, voc,
                                    input_sentence)
            output_words[:] = [x for x in output_words if not (x == 'EOS' or
                                                               x == 'PAD')]
            print('Bot:', ' '.join(output_words))
        except KeyError:
            print("Error: Encountered unknown word.")
