"""
This module handles reading the corpus and parsing into Voc for use within the
chatbot.
"""
from typing import List
from io import open
import os
import itertools

import torch

from util import config, normalize_string
from vocab import Voc, EOS_TOKEN, PAD_TOKEN

MAX_LENGTH = int(config()['DEFAULT']['MAX_LENGTH'])
DATA_PATH = config()['data']['data_path']


def read_vocs(datafile: str, corpus_name: str) -> (Voc, List[List[str]]):
    """
    Read query/response pairs and return voc object.
    """
    print(f"Reading lines from {datafile}...")

    pairs = []
    with open(datafile, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.split('+++$+++')
            pairs.append([
                normalize_string(line[0]),
                normalize_string(line[1])
            ])

    voc = Voc(corpus_name)
    return voc, pairs


def filter_pair(pair: List[str]) -> bool:
    """
    Returns True iff both sentences in a pair 'pair' are under the MAX_LENGTH
    threshold.
    """
    return len(pair) == 2 and max(len(pair[0].split(' ')),
                                  len(pair[1].split(' '))) <= MAX_LENGTH


def filter_pairs(pairs: List[List[str]]) -> List[List[str]]:
    """
    Filter out pairs by using filter_pair condition.
    """
    return [pair for pair in pairs if filter_pair(pair)]


def load_prepare_data(corpus_name: str, datafile: str, save_dir: str = "") \
        -> (Voc, List[List[str]]):
    """
    Loads a corpus from file and creates a Voc object to hold information for
    the corpus.
    """
    print("Start preparing training data...")
    voc, pairs = read_vocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])
    if len(save_dir) > 0:
        with open(os.path.join(os.path.dirname(__file__), save_dir, voc.name),
                  'w', encoding='utf-8') as vocab_file:
            for [word, _] in voc.word2count:
                vocab_file.write('%s\n' % word)

    print("Counted words:", voc.num_words)
    return voc, pairs


# save_dir = os.path.join("data", "save")
# voc, pairs = load_prepare_data()

def indeces_from_sentence(voc: Voc, sentence: str) -> List[str]:
    return [voc.word2index[word if word in voc.word2index else "UNK"] \
            for word in sentence.split(' ')] + [EOS_TOKEN]


def zero_padding(l: list, fill_value: int = PAD_TOKEN) -> List[int]:
    return list(itertools.zip_longest(*l, fillvalue=fill_value))


def binary_matrix(l: list, value: int = PAD_TOKEN):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def input_var(l: list, voc: Voc) -> (torch.LongTensor, torch.Tensor):
    """Input tensors for training data"""
    indeces_batch = [indeces_from_sentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indeces) for indeces in indeces_batch])
    pad_list = zero_padding(indeces_batch)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths


def output_var(l: list, voc: Voc) -> (torch.LongTensor, torch.BoolTensor, int):
    """Output tensors and mask for training data"""
    indeces_batch = [indeces_from_sentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indeces) for indeces in indeces_batch])
    pad_list = zero_padding(indeces_batch)
    mask = binary_matrix(pad_list)
    mask = torch.BoolTensor(mask)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, mask, max_target_len


def batch_to_train_data(voc: Voc, pair_batch: List[List[str]]) \
        -> (list, torch.Tensor, torch.LongTensor, torch.BoolTensor, int):
    """Turn sentence pair batches into training data"""
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = input_var(input_batch, voc)
    output, mask, max_target_len = output_var(output_batch, voc)
    return inp, lengths, output, mask, max_target_len
