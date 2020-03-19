"""
This module contains the class for Voc which is the vocabulary handler for the
chatbot.
"""
from util import config

vocab_size = int(config()['data']['vocab_size'])

# Default word tokens
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3


class Voc:
    """Class for holding the vocabulary for a set of data"""
    def __init__(self, name: str):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS",
                           UNK_TOKEN: "UNK"}
        self.num_words = len(self.index2word)

    def add_sentence(self, sentence: str) -> None:
        """
        Takes in a sentence and adds the words of the sentence into the vocab.
        """
        for word in sentence.split(' '):
            if word not in ["UNK", "SOS", "EOS", "PAD"]:
                word = word.lower()
            self.add_word(word)

    def add_word(self, word: str) -> None:
        """
        Add a word to the vocabulary, if already exists, increase the count for
        that word.
        """
        if self.num_words >= vocab_size - 1:
            word = "UNK"

        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count) -> None:
        """
        Trims the set and remove words below min_count.
        """
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for word, count in self.word2count.items():
            if count >= min_count:
                keep_words.append(word)
