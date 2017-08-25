'''
This file is modified from https://raw.githubusercontent.com/spro/practical-pytorch/master/char-rnn-classification/data.py
'''

import torch
import glob
import unicodedata
import string

all_letters = string.printable
n_letters = len(all_letters)

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('../data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# Turn multiple lines into a <bathc_size x line_length x n_letters>
def linesToTensor(lines):
    line_length = max([ len(line) for line in lines]) 
    tensor = torch.zeros(len(lines), line_length, n_letters)
    for b, line in enumerate(lines): 
        for li, letter in enumerate(line):
            tensor[b][li + line_length - len(line)][letterToIndex(letter)] = 1

    return tensor
    
def word_linesToTensor(lines, corpus):  
    line_length = max([ len(line.split()) for line in lines ])
    tensor = torch.zeros(len(lines), line_length, len(corpus))

    for b, line in enumerate(lines):
        for li, word in enumerate(line.split()):
            tensor[b][li + line_length - len(line.split())][ corpus.index_word(word) ] = 1
    
    return tensor
