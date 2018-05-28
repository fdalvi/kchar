import argparse
import codecs
import cPickle as pickle
import json
import numpy as np
import os
import re

from math import exp
from model.LSTMCNN import LSTMCNN_with_mask
from util.BatchLoaderUnk import Tokens, encoding # needed by pickle.load()

def vocab_unpack(vocab):
    return vocab['idx2word'], vocab['word2idx'][()], vocab['idx2char'], vocab['char2idx'][()]

class Vocabulary:
    def __init__(self, tokens, vocab_file, max_word_l=65):
        self.tokens = tokens
        self.max_word_l = max_word_l
        self.prog = re.compile('\s+')

        print('loading vocabulary file...')
        vocab_mapping = np.load(vocab_file)
        self.idx2word, self.word2idx, self.idx2char, self.char2idx = vocab_unpack(vocab_mapping)
        self.vocab_size = len(self.idx2word)
        print 'Word vocab size: %d, Char vocab size: %d' % (len(self.idx2word), len(self.idx2char))
        self.word_vocab_size = len(self.idx2word)
        self.char_vocab_size = len(self.idx2char)

    def index(self, word):
        if word[0] == self.tokens.UNK and len(word) > 1: # unk token with character info available
            word = word[2:]
            w = self.word2idx[self.tokens.UNK]
        else:
            w = self.word2idx[word] if word in self.word2idx else self.word2idx[self.tokens.UNK]
            
        c = np.zeros(self.max_word_l, dtype='int32')
        chars = [self.char2idx[self.tokens.START]] # start-of-word symbol
        chars += [self.char2idx[char] for char in word if char in self.char2idx]
        chars.append(self.char2idx[self.tokens.END]) # end-of-word symbol
        if len(chars) >= self.max_word_l:
            chars[self.max_word_l-1] = self.char2idx[self.tokens.END]
            c = chars[:self.max_word_l]
        else:
            c[:len(chars)] = chars
            
        return w, c

    def get_input(self, line):
        output_words = []
        output_chars = []

        line = line.replace('<unk>', self.tokens.UNK)  # replace unk with a single character
        line = line.replace(self.tokens.START, '')  # start-of-word token is reserved
        line = line.replace(self.tokens.END, '')  # end-of-word token is reserved
        words = self.prog.split(line)
        for rword in filter(None, words):
            w, c = self.index(rword)
            output_words.append(w)
            output_chars.append(c)
        if self.tokens.EOS != '':   # PTB does not have <eos> so we add a character for <eos> tokens
            w, c = self.index(self.tokens.EOS)   # other datasets don't need this
            output_words.append(w)
            output_chars.append(c)

        words = np.array(output_words[-1:] + output_words[:-1], dtype='int32')
        chars = np.array(output_chars[-1:] + output_chars[:-1], dtype='int32')[:, np.newaxis, :]
        output = np.array(output_words, dtype='int32')[:, np.newaxis, np.newaxis]
        return ({'word':words, 'chars':chars}, output)


class evaluator:
    def __init__(self, name, vocabulary, mask):
        self.opt = pickle.load(open('{}.pkl'.format(name), "rb"))
        self.opt.batch_size = 1
        self.opt.seq_length = 1
        self.reader = Vocabulary(self.opt.tokens, vocabulary, max_word_l=self.opt.max_word_l)
        self.model = LSTMCNN_with_mask(self.opt, mask)
        self.model.load_weights('{}.h5'.format(name))
        print self.model.summary()
        
        self.state_mean = None

    def logprob(self, line):
        x, y = self.reader.get_input(line)
        nwords = len(y)
        if self.state_mean is not None:
            self.model.set_states_value(self.state_mean)

        lprob = self.model.evaluate(x, y, batch_size=1, verbose=0)
        return lprob, nwords

def main(name, vocabulary, text, mask_file):

    with open(mask_file) as fp:
        mask_file_content = "\n".join(fp.readlines())
        mask_file_content = re.sub(r'//.*\n', '\n', mask_file_content)
        mask = json.loads(mask_file_content)

    ev = evaluator(name, vocabulary, mask)

    f = codecs.open(text, 'r', encoding)

    lp = 0;
    nw = 0;
    nl = 0;

    count = 0
    for line in f:
        lprob, nwords = ev.logprob(line)
    	lp += lprob*nwords
    	nw += nwords
    	nl += 1
    	print "Perplexity = ", exp(lp/nw), "\t(", nl, ")"

    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--vocabulary', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--mask_file', type=str, required=True)

    args = parser.parse_args()

    main(args.model, args.vocabulary, args.text, args.mask_file)
