import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import numpy as np
import sys
import copy
from log_ops import LogOps

"""
Chiu2015 Bi-directional LSTM CNN
"""
CAPITALIZATION_DIM = 5


class Network(nn.Module):

    def __init__(self, vocab_size, number_tags, hidden_size=275, embedding_dim=50,
                 retrain_embeddings=True, gpu=False, include_cap=False, dropout=0.68):
        super(Network, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.gpu = gpu
        self.include_cap = include_cap
        self.number_tags = number_tags
        self.hidden_size = hidden_size
        self.logops = LogOps(gpu)

        if self.gpu and torch.cuda.is_available():
            self.long_type = torch.cuda.LongTensor
            self.float_type = torch.cuda.FloatTensor
        else:
            self.long_type = torch.LongTensor
            self.float_type = torch.FloatTensor

        self.transitions = nn.Parameter(torch.zeros(number_tags, number_tags))
        self.initial_transitions = nn.Parameter(torch.zeros(number_tags))

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.requires_grad = retrain_embeddings

        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=hidden_size, num_layers=1, batch_first=True,
                            dropout=dropout, bidirectional=True)
        # self.lstm_forward = nn.LSTM(input_size=self.embedding_dim,
        #                             hidden_size=hidden_size, num_layers=1, batch_first=True,
        #                             dropout=dropout)
        # self.lstm_backward = nn.LSTM(input_size=self.embedding_dim,
        #                              hidden_size=hidden_size, num_layers=1, batch_first=True,
        #                              dropout=dropout)

        self.linear = nn.Linear(hidden_size, number_tags)
        # self.linear_forward = nn.Linear(hidden_size, number_tags)
        # self.linear_backward = nn.Linear(hidden_size, number_tags)
        self.softmax = nn.LogSoftmax(dim=1)

    """
    Input is LongTensor(N,W)
    N - mini batch size
    W - number of indices to extract (sequence length)
    """
    def forward(self, input):
        # input_list = input.data.tolist()
        # for l in input_list:
        #     l = l.reverse()
        # input_backward = Variable(self.long_type(input_list))
        #
        # f_emb = self.embeddings(input)
        # b_emb = self.embeddings(input_backward)
        #
        # o_f = self.lstm_forward(f_emb)
        # o_b = self.lstm_backward(b_emb)
        #
        # # other outputs are hidden states and cell states
        # o_f = o_f[0]
        # o_b = o_b[0]
        #
        # # reverse backward lstm outputs
        # reversed_output = torch.zeros_like(o_b)
        # for batch_num, batch in enumerate(o_b):
        #     for s_num, s in enumerate(o_b[batch_num]):
        #         reversed_output[batch_num, s_num] = o_b[batch_num, o_b.shape[1]-1-s_num]
        #
        # # (batch_size,seq_len,num_tags)
        # output_tensor = Variable(self.float_type(input.shape[0],
        #                         input.shape[1], self.number_tags))
        #
        # for seq, (step_forward, step_backward) in enumerate(zip(o_f, o_b)):
        #     o_f = self.linear_forward(step_forward)
        #     o_b = self.linear_backward(step_backward)
        #     o_f = self.softmax(o_f)
        #     o_b = self.softmax(o_b)
        #     o = o_f+o_b
        #     output_tensor[seq] = o
        # b = output_tensor != output_tensor
        # assert not b.any()  # LSTM OUTPUT IS NAN
        # return output_tensor

        emb = self.embeddings(input)
        o = self.lstm(emb)
        # (batch, seq_len, hidden_size * num_directions)
        o = o[0]
        f_o = o[:, :, :self.hidden_size]
        b_o = o[:, :, self.hidden_size:]

        f_o = self.linear(f_o)
        b_o = self.linear(b_o)

        f_o = self.softmax(f_o)
        b_o = self.softmax(b_o)

        o = f_o+b_o

        return o
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

    def get_train_tensors(self, words, tags=[], word2id=None, tag2id=None, batch_size=1):
        assert len(words) == len(tags)
        assert len(words[0]) == batch_size

        word_tensor = self.long_type(batch_size, len(words))
        tag_tensor = self.long_type(batch_size, len(words))

        for pos, (word_list, tag_list) in enumerate(zip(words, tags)):
            for batch, (word, tag) in enumerate(zip(word_list, tag_list)):
                word_tensor[batch][pos] = word2id(word)
                tag_tensor[batch][pos] = tag2id(tag)

        return word_tensor, tag_tensor  # ,cap_tensor
