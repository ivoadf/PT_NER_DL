import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import numpy as np
import sys
import copy
from log_ops import LogOps
from models.mellon_char_cnn import CharCNN

"""
Carnegie Mellon Bi-directional LSTM CNN, paper = Ma2016
"""
CHAR_REP_SIZE = 30
DROPOUT = 0.5

class Network(nn.Module):

    def __init__(self, vocab_size, number_tags, number_chars, hidden_size=200,
                 batch_size=1, embedding_dim=50, retrain_embeddings=True,
                 gpu=False):
        super(Network, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.gpu = gpu
        self.number_tags = number_tags
        self.number_chars = number_chars
        self.logops = LogOps(gpu)

        if self.gpu and torch.cuda.is_available():
            self.long_type = torch.cuda.LongTensor
            self.float_type = torch.cuda.FloatTensor
        else:
            self.long_type = torch.LongTensor
            self.float_type = torch.FloatTensor

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.requires_grad = retrain_embeddings
        self.char_cnn = CharCNN(self.number_chars, gpu=self.gpu)

        self.dropout = nn.Dropout(p=DROPOUT)

        self.lstm_forward = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size,
                                    num_layers=1, batch_first=True, dropout=DROPOUT)
        
        # self.lstm_backward = nn.LSTM(input_size=self.embedding_dim,
        #                              hidden_size=hidden_size, num_layers=1, batch_first=True,
        #                              dropout=dropout)
        #
        # self.linear_forward = nn.Linear(hidden_size, number_tags)
        # self.linear_backward = nn.Linear(hidden_size, number_tags)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, words_var, words, char_to_id):
        assert len(words_var) == self.batch_size
        assert len(words_var[0]) == len(words)

        char_representations = Variable(self.float_type(len(words_var), len(words_var[0]), CHAR_REP_SIZE))
        for seq_id, batch in enumerate(words):
            for batch_id, word in enumerate(batch):
                char_cnn_input = self.char_cnn.word_to_char_var(word, char_to_id)
                word_rep = self.char_cnn(char_cnn_input)
                char_representations[batch_id, seq_id, :] = word_rep
        return char_representations

        word_emb = self.embeddings =

    """
    Score sequence

    seq_preds: output of lstm (batch,seqlen,num_tags)
    tag_ids: (batch_size,sequence_lenght) tags ids to evaluate
    """
    def score_of_sequence(self, seq_preds, tag_ids):
        scores = Variable(self.float_type(len(seq_preds)))
        for i, (lstm_out, tags) in enumerate(zip(seq_preds, tag_ids)):
            score = self.initial_transitions[tags[0]] + lstm_out[0, tags[0]]
            previous_tag_id = tags[0]
            for word_scores, tag in zip(lstm_out[1:], tags[1:]):
                score += self.transitions[previous_tag_id, tag]+word_scores[tag]
                previous_tag_id = tag
            scores[i] = score
        return scores

    """
    Sentence level log likelihood
    input: output of lstm (batch,seqlen,num_tags)
    real_tags: (batch_size,sequence_length) real tags to compare the results to
    """
    def sentence_level_log_likelihood(self, lstm_out, real_tags):
        sequence_length = lstm_out.shape[1]

        S = self.score_of_sequence(lstm_out, real_tags)
        scores = Variable(self.float_type(len(S)))

        for batch_num, batch_score in enumerate(S):
            backward = self.float_type(sequence_length, self.number_tags).zero_() - float('inf')
            backward = Variable(backward)

            backward[sequence_length-1, :] = lstm_out[batch_num, sequence_length-1, :]

            for pos in range(sequence_length-2, -1, -1):
                for state in range(self.number_tags):
                    backward[pos, state] = \
                        self.logops.logsumexp(backward[pos+1, :] + self.transitions[:, state]) +\
                        lstm_out[batch_num, pos+1, state]
            log_likelihood = self.logops.logsumexp(backward[0, :] + self.initial_transitions) +\
                lstm_out[batch_num, 0, :]
            log_likelihood = self.logops.logsumexp(log_likelihood)
            scores[batch_num] = batch_score - log_likelihood
        return torch.neg(torch.mean(scores))

    """
    Viterbi algorithm to find the tag sequence that maximizes the score

    emission_scores = (sequence_length,number_tags)
    """
    def viterbi_decode(self, emission_scores):
        emission_scores = emission_scores.data
        sequence_length = len(emission_scores)

        viterbi_scores = self.float_type(sequence_length, self.number_tags).zero_() - float('inf')

        viterbi_paths = self.long_type(sequence_length, self.number_tags)
        nn.init.constant(viterbi_paths, -1)

        best_path = self.long_type(sequence_length)

        viterbi_scores[0, :] = self.initial_transitions.data

        transitions = self.transitions.data

        for pos in range(1, sequence_length, 1):
            for state in range(self.number_tags):
                viterbi_scores[pos, state] = torch.max(viterbi_scores[pos-1, :]
                                                       + transitions[state, :])
                viterbi_scores[pos, state] += emission_scores[pos, state]
                _, indexes = \
                    torch.topk(viterbi_scores[pos-1, :] + transitions[state, :], 1)
                viterbi_paths[pos, state] = indexes[0]
        best_score = torch.max(viterbi_scores[sequence_length-1, :]
                               + emission_scores[sequence_length-1, :])

        _, indexes = torch.topk(viterbi_scores[sequence_length-1, :], 1)
        best_path[sequence_length-1] = indexes[0]

        for pos in range(sequence_length-2, -1, -1):
            best_path[pos] = viterbi_paths[pos+1, best_path[pos+1]]

        return best_score, best_path

    def set_embeddings_weights(self, emb_array):
        assert len(emb_array) == self.vocab_size and len(emb_array[0]) == self.embedding_dim
        weights = torch.FloatTensor(emb_array)
        if self.gpu and torch.cuda.is_available():
            weights.cuda()
        self.embeddings.weight.data.copy_(weights)

    def get_train_tensors(self, words, tags=[], word2id=None, tag2id=None):
        assert len(words) == len(tags)
        assert len(words[0]) == self.batch_size

        word_tensor = self.long_type(self.batch_size, len(words))
        tag_tensor = self.long_type(self.batch_size, len(words))

        for pos, (word_list, tag_list) in enumerate(zip(words, tags)):
            for batch, (word, tag) in enumerate(zip(word_list, tag_list)):
                word_tensor[batch][pos] = word2id(word)
                tag_tensor[batch][pos] = tag2id(tag)

        return Variable(word_tensor), Variable(tag_tensor)
