"""
This module reads the config file for the corpus and creates a corresponding
corpus file.
"""
import os
from util import config

CORPUS_FILES = [
    config()['data']['in_data'].split(','),
    config()['data']['out_data'].split(',')
]
print('input files: {!s}'.format(CORPUS_FILES[0]))
print('output files: {!s}'.format(CORPUS_FILES[0]))


def build_set(name):
    """Builds a full dataset with name 'name' and saves"""
    num_samples = 0
    samples = int(config()['data']['samples'])
    for i in range(min(len(CORPUS_FILES[0]), len(CORPUS_FILES[1]))):
        with open(os.path.join(
                os.path.dirname(__file__),
                config()['data']['data_path'], name + '.train'), 'a',
                encoding='utf-8') \
                as corpus_file:

            in_data = read_file(CORPUS_FILES[0][i])
            out_data = read_file(CORPUS_FILES[1][i])
            for j in range(
                    min([len(in_data), len(out_data), samples - num_samples])):
                corpus_file.write('%s+++$+++%s\n' % (in_data[j], out_data[j]))


def read_file(file_name):
    """Reads a file thing"""
    lines = []
    with open(os.path.join(os.path.dirname(__file__),
                           config()['data']['data_path'], file_name), 'r',
              encoding='utf-8') as f:
        line = f.readline().strip()
        while line:
            lines.append(line.lower())
            line = f.readline().strip()
    return lines
