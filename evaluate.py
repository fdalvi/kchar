import codecs
import numpy as np
import re
import argparse
import cPickle as pickle
from model.LSTMCNN import LSTMCNN
from model.LSTMCNN import LSTMCNN_print
from util.BatchLoaderUnk import Tokens, encoding # needed by pickle.load()
from math import exp
import os

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
    def __init__(self, name, vocabulary, init, extract, layer):
        self.opt = pickle.load(open('{}.pkl'.format(name), "rb"))
        self.opt.batch_size = 1
        self.opt.seq_length = 1
        self.reader = Vocabulary(self.opt.tokens, vocabulary, max_word_l=self.opt.max_word_l)
        if extract:
            self.model = LSTMCNN_print(self.opt, extract, layer)
            print self.model.summary()
            self.model.load_weights('{}.h5'.format(name), by_name=True)
        else:
            self.model = LSTMCNN(self.opt)
            self.model.load_weights('{}.h5'.format(name))
            print self.model.summary()
        if init:
            self.state_mean = np.load(init)
        else:
            self.state_mean = None

    def logprob(self, line):
        x, y = self.reader.get_input(line)
        nwords = len(y)
        if self.state_mean is not None:
            self.model.set_states_value(self.state_mean)
        return self.model.evaluate(x, y, batch_size=1, verbose=0), nwords

    def get_embedding(self, line):
        x, y = self.reader.get_input(line)
        if self.state_mean is not None:
            self.model.set_states_value(self.state_mean)
        return self.model.predict(x, batch_size=1, verbose=0)

def main(name, vocabulary, init, text, calc, extract, layer):

    ev = evaluator(name, vocabulary, None if calc else init, extract, layer)

    if extract:
        outputFile = os.path.basename(text) + "_" + extract + "_" + str(layer) + "_output"
        em = codecs.open(outputFile, 'w', encoding='utf-8')

    f = codecs.open(text, 'r', encoding)
    num_lines = sum(len(line.split(" ")) for line in f)

    f.seek(0)
    if calc:
        lp = 0;
        nw = 0;
        nl = 0;
        state_sum = [np.zeros_like(a) for a in ev.model.state_updates_value]
        for line in f:
            lprob, nwords = ev.logprob(line)
            lp += lprob*nwords
            nw += nwords
            for ssum, update in zip(state_sum, ev.model.state_updates_value):
                ssum += update
            nl += 1
            print "Perplexity = ", exp(lp/nw), "\t(", nl, ")", ssum[0][0]/nl

        state_mean = [a/nl for a in state_sum]
        np.save(init, state_mean)
    else:
        lp = 0;
        nw = 0;
        nl = 0;
        f.seek(0)

        count = 0
        for line in f:
            if extract:
                s_arr = []
                line = line.strip()
                words = line.split(" ")

                s = ev.get_embedding(line)

                if count == 0:
                    emb_size = (s.shape)[2] # embedding size
                    em.write(str(num_lines) + " " + str(emb_size) + "\n")

                for word_idx in range(1, s.shape[0]): # ignoring start of symbol
                    s_arr = map(str, s[word_idx,:,:].flatten().tolist()) # convert numpy array to list and convert float to string
                    # taking the embedding of the first word only in case the input consists of more than one word per line
                    em.write(words[word_idx-1] + " " + " ".join(s_arr) + "\n")

                count += 1
            else:
                lprob, nwords = ev.logprob(line)
            	lp += lprob*nwords
            	nw += nwords
            	nl += 1
            	print "Perplexity = ", exp(lp/nw), "\t(", nl, ")"
    if extract:
        em.close()
        print "%s file written to the working directory" % (outputFile)
        
    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--vocabulary', type=str)
    parser.add_argument('--init', type=str)
    parser.add_argument('--text', type=str)
    parser.add_argument('--calc', action='store_true', default=False)
    parser.add_argument('--extract', choices=['embedding', 'highway', 'feedforward', 'feedforward1', 'rnn'])
    parser.add_argument('--layer', type=int, default=0) # for rnn, which layer to extract

    args = parser.parse_args()

    main(args.model, args.vocabulary, args.init, args.text, args.calc, args.extract, args.layer)
