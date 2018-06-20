from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-CRF model for NER.
"""

import sys
import os

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from neuronlp2.io import get_logger, conll03_data, CoNLL03Writer
from neuronlp2.models import BiRecurrentConvCRF, BiVarRecurrentConvCRF
from neuronlp2 import utils

uid = uuid.uuid4().hex[:6]


def evaluate(output_file):
    score_file = "tmp/score_%s" % str(uid)
    os.system("examples/eval/conll03eval.v2 < %s > %s" % (output_file, score_file))
    with open(score_file, 'r') as fin:
        fin.readline()
        line = fin.readline()
        fields = line.split(";")
        acc = float(fields[0].split(":")[1].strip()[:-1])
        precision = float(fields[1].split(":")[1].strip()[:-1])
        recall = float(fields[2].split(":")[1].strip()[:-1])
        f1 = float(fields[3].split(":")[1].strip())
    return acc, precision, recall, f1


def get_entities(words, tags):
    entities = []
    entity = []
    state = 1
    for i, tag in enumerate(tags):
        # transitions from state 1
        if state == 0:
            if tag[0] == 'B':
                state = 1
                entity.append(words[i])
            continue
        if state == 1:
            if tag[0] == 'I':
                entity.append(words[i])
            elif tag[0] == 'O' or tag[0] == 'B':
                if len(entity) > 0:
                    entities.append(' '.join(entity))
                entity = []
                state = 0
            if tag[0] == 'B':
                entity = [words[i]]
            continue
    return entities

class MellonBoot():
    def __init__(self, train, test, embeddings_filename, batch_size=1):
        self.train_path = train
        self.test_path = test
        self.mode = 'LSTM'
        self.dropout = 'std'
        self.num_epochs = 1
        self.batch_size = batch_size
        self.hidden_size = 256
        self.num_filters = 30
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.decay_rate = 0.05
        self.gamma = 0.0
        self.schedule = 1
        self.p_rnn = tuple([0.33, 0.5])
        self.p_in = 0.33
        self.p_out = 0.5
        self.unk_replace = 0.0
        self.bigram = True
        self.embedding = 'glove'
        self.logger = get_logger("NERCRF")
        self.char_dim = 30
        self.window = 3
        self.num_layers = 1
        self.tag_space = 128
        self.initializer = nn.init.xavier_uniform

        self.use_gpu = torch.cuda.is_available()

        self.embedd_dict, self.embedd_dim = utils.load_embedding_dict(
            self.embedding, embeddings_filename)
        self.word_alphabet, self.char_alphabet, self.pos_alphabet, \
        self.chunk_alphabet, self.ner_alphabet = conll03_data.create_alphabets("data/alphabets/ner_crf/", self.train_path, data_paths=[self.test_path],
        embedd_dict=self.embedd_dict, max_vocabulary_size=50000)
        self.word_table = self.construct_word_embedding_table()

        self.logger.info("Word Alphabet Size: %d" % self.word_alphabet.size())
        self.logger.info("Character Alphabet Size: %d" % self.char_alphabet.size())
        self.logger.info("POS Alphabet Size: %d" % self.pos_alphabet.size())
        self.logger.info("Chunk Alphabet Size: %d" % self.chunk_alphabet.size())
        self.logger.info("NER Alphabet Size: %d" % self.ner_alphabet.size())
        self.num_labels = self.ner_alphabet.size()

        self.data_test = conll03_data.read_data_to_variable(self.test_path, self.word_alphabet,
                                                            self.char_alphabet, self.pos_alphabet,
                                                            self.chunk_alphabet, self.ner_alphabet,
                                                            use_gpu=self.use_gpu,
                                                            volatile=True)
        self.writer = CoNLL03Writer(self.word_alphabet, self.char_alphabet, self.pos_alphabet,
                                    self.chunk_alphabet, self.ner_alphabet)

    def train(self):
        network = BiRecurrentConvCRF(self.embedd_dim, self.word_alphabet.size(), self.char_dim,
                                     self.char_alphabet.size(), self.num_filters, self.window,
                                     self.mode, self.hidden_size, self.num_layers, self.num_labels,
                                     tag_space=self.tag_space, embedd_word=self.word_table,
                                     p_in=self.p_in, p_out=self.p_out, p_rnn=self.p_rnn,
                                     bigram=self.bigram, initializer=self.initializer)

        if self.use_gpu:
            network.cuda()

        optim = SGD(network.parameters(), lr=self.learning_rate, momentum=self.momentum,
                    weight_decay=self.gamma, nesterov=True)

        self.logger.info("Network: %s, num_layer=%d, hidden=%d, filter=%d, tag_space=%d, crf=%s" % (self.mode, self.num_layers, self.hidden_size, self.num_filters, self.tag_space, 'bigram' if self.bigram else 'unigram'))
        self.logger.info("training: l2: %f, (#training data: %d, batch: %d, unk replace: %.2f)" % (self.gamma, self.num_data, self.batch_size, self.unk_replace))
        self.logger.info("dropout(in, out, rnn): (%.2f, %.2f, %s)" % (self.p_in, self.p_out, self.p_rnn))

        num_batches = self.num_data / self.batch_size + 1

        dev_f1 = 0.0
        dev_acc = 0.0
        dev_precision = 0.0
        dev_recall = 0.0
        test_f1 = 0.0
        test_acc = 0.0
        test_precision = 0.0
        test_recall = 0.0
        best_epoch = 0
        for epoch in range(1, self.num_epochs + 1):
            print('Epoch %d (%s(%s), learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (epoch, self.mode, self.dropout, self.learning_rate, self.decay_rate, self.schedule))
            train_err = 0.
            train_total = 0.

            start_time = time.time()
            num_back = 0
            network.train()
            for batch in range(1, num_batches + 1):
                word, char, _, _, labels, masks, lengths = conll03_data.get_batch_variable(self.data_train, self.batch_size, unk_replace=self.unk_replace)
                optim.zero_grad()
                loss = network.loss(word, char, labels, mask=masks)
                loss.backward()
                optim.step()

                num_inst = word.size(0)
                train_err += loss.data[0] * num_inst
                train_total += num_inst

                time_ave = (time.time() - start_time) / batch
                time_left = (num_batches - batch) * time_ave

                # update log
                if batch % 100 == 0:
                    sys.stdout.write("\b" * num_back)
                    sys.stdout.write(" " * num_back)
                    sys.stdout.write("\b" * num_back)
                    log_info = 'train: %d/%d loss: %.4f, time left (estimated): %.2fs' % (batch, num_batches, train_err / train_total, time_left)
                    sys.stdout.write(log_info)
                    sys.stdout.flush()
                    num_back = len(log_info)

            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            print('train: %d loss: %.4f, time: %.2fs' % (num_batches, train_err / train_total, time.time() - start_time))

            if epoch % self.schedule == 0:
                lr = self.learning_rate / (1.0 + epoch * self.decay_rate)
                optim = SGD(network.parameters(), lr=lr, momentum=self.momentum, weight_decay=self.gamma, nesterov=True)
        self.network = network

    def test(self):
        self.network.eval()
        # evaluate on test data when better performance detected
        tmp_filename = 'tmp/%s_test' % (str(uid))
        self.writer.start(tmp_filename)

        for batch in conll03_data.iterate_batch_variable(self.data_test, self.batch_size):
            word, char, pos, chunk, labels, masks, lengths = batch
            preds, _ = self.network.decode(word, char, target=labels, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
            self.writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(), preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
        self.writer.close()
        test_acc, test_precision, test_recall, test_f1 = evaluate(tmp_filename)
        return test_acc, test_precision, test_recall, test_f1

    def test_file(self, file_path):
        self.network.eval()
        tmp_filename = 'tmp/%s_test' % (str(uid))
        self.writer.start(tmp_filename)

        self.data_small_test = conll03_data.read_data_to_variable(file_path, self.word_alphabet,
                                                            self.char_alphabet, self.pos_alphabet,
                                                            self.chunk_alphabet, self.ner_alphabet,
                                                            use_gpu=self.use_gpu,
                                                            volatile=True)

        for batch in conll03_data.iterate_batch_variable(self.data_small_test, self.batch_size):
            word, char, pos, chunk, labels, masks, lengths = batch
            preds, _ = self.network.decode(word, char, target=labels, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
            self.writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(), preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
        self.writer.close()
        test_acc, test_precision, test_recall, test_f1 = evaluate(tmp_filename)
        return test_acc, test_precision, test_recall, test_f1

    def name_list(self):
        num_batches = self.num_data / self.batch_size + 1
        total_name_dict = {}
        self.network.eval()
        # evaluate on test data when better performance detected
        tmp_filename = 'tmp/%s_name_list' % (str(uid))
        self.writer.start(tmp_filename)

        for batch in range(1, num_batches + 1):
            word, char, pos, chunk, labels, masks, lengths = conll03_data.get_batch_variable(self.data_train, self.batch_size, unk_replace=self.unk_replace)
            preds, _ = self.network.decode(word, char, target=labels, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
            self.writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(), preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
        self.writer.close()
        #load file and extract namelist
        words = []
        tag_list = []
        names = []
        f = open(tmp_filename, "r")
        for line in f:
            if line == '\n':
                names += get_entities(words, tag_list)
                words = []
                tag_list = []
            else:
                line_s = line.split()
                words.append(line_s[1])
                tag_list.append(line_s[5])
        names += get_entities(words, tag_list)
        for name in names:
            if name in total_name_dict:
                total_name_dict[name] += 1
            else:
                total_name_dict[name] = 1
        return set([name for name, num_occ in total_name_dict.items() if num_occ >= 4])

    def set_train_dataset(self, train_filename):
        self.train_path = train_filename
        self.data_train = conll03_data.read_data_to_variable(self.train_path, self.word_alphabet,
                                                             self.char_alphabet, self.pos_alphabet,
                                                             self.chunk_alphabet, self.ner_alphabet,
                                                             use_gpu=self.use_gpu)
        self.num_data = sum(self.data_train[1])

    def save_model(self, save_path, bootstrapping_iteration):
        save_name = "MELLON_epochs_{}_bootit_{}".format(
            self.num_epochs,
            bootstrapping_iteration)
        torch.save(self.network.state_dict(), os.path.join(save_path, save_name+'.model'))

    def construct_word_embedding_table(self):
        scale = np.sqrt(3.0 / self.embedd_dim)
        table = np.empty([self.word_alphabet.size(), self.embedd_dim], dtype=np.float32)
        table[conll03_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, self.embedd_dim]).astype(np.float32)
        oov = 0
        for word, index in self.word_alphabet.items():
            if word in self.embedd_dict:
                embedding = self.embedd_dict[word]
            elif word.lower() in self.embedd_dict:
                embedding = self.embedd_dict[word.lower()]
            else:
                embedding = np.random.uniform(-scale, scale, [1, self.embedd_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('oov: %d' % oov)
        return torch.from_numpy(table)
