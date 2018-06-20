import torch.nn as nn
import torch
from architectures.utils.helper import capitalization_analysis
import sys
from torch.autograd import Variable
from architectures.utils.log_ops import LogOps
"""
Window approach, concatenate representations of all words in window.
Following 2 Linear layers with a hard tanh in between.
"""
CAPITALIZATION_DIM = 5
FEATURE_DIM = 5


class NN(nn.Module):

    def __init__(self, vocab_size, number_tags, number_suffix, hidden_size=300,
                 batch_size=1, window_size=2, embedding_dim=200, retrain_embeddings=False,
                 gpu=False, include_cap=False, activation='hardtanh'):
        super(NN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.batch_size = batch_size
        self.gpu = gpu
        self.include_cap = include_cap
        self.number_tags = number_tags
        self.logops = LogOps(gpu)

        if self.gpu and torch.cuda.is_available():
            self.long_type = torch.cuda.LongTensor
            self.float_type = torch.cuda.FloatTensor
        else:
            self.long_type = torch.LongTensor
            self.float_type = torch.FloatTensor

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.requires_grad = retrain_embeddings

        self.transitions = nn.Parameter(torch.zeros(number_tags, number_tags))
        self.initial_transitions = nn.Parameter(torch.zeros(number_tags))

        if include_cap:
            self.cap_embeddings = nn.Embedding(CAPITALIZATION_DIM, FEATURE_DIM)
            self.suf_embeddings = nn.Embedding(number_suffix, FEATURE_DIM)
            embedding_dim += 2*FEATURE_DIM
        self.i2h = nn.Linear((window_size*2+1)*embedding_dim, hidden_size)
        if activation == 'relu':
            self.activation_f = nn.ReLU()
        elif activation == 'tanh':
            self.activation_f = nn.Tanh()
        elif activation == 'hardtanh':
            self.activation_f = nn.Hardtanh()
        self.h2o = nn.Linear(hidden_size, number_tags)

    """
    Input is LongTensor(N,W)
    N - mini batch size
    W - number of indices to extract (window_size)
    """
    def forward(self, input, cap_tensor, suf_tensor):
        o = self.embeddings(input)
        if self.include_cap:  # concatenate capitalization/suffix info
            cap_emb = self.cap_embeddings(cap_tensor)
            suf_emb = self.suf_embeddings(suf_tensor)
            o = torch.cat((o, cap_emb), 2)
            o = torch.cat((o, suf_emb), 2)
        o = o.view(self.batch_size, -1)
        o = self.i2h(o)
        o = self.activation_f(o)
        o = self.h2o(o)
        return o

    """
    Score sequence

    seq_preds: output of network (batch,seqlen,num_tags)
    tag_ids: (batch_size,sequence_lenght) tags ids to evaluate
    """
    def score_of_sequence(self, seq_preds, tag_ids):
        scores = Variable(self.float_type(len(seq_preds)))

        for i, (network_out, tags) in enumerate(zip(seq_preds, tag_ids)):
            score = self.initial_transitions[tags[0]] + network_out[0, tags[0]]
            previous_tag_id = tags[0]
            for word_scores, tag in zip(network_out[1:], tags[1:]):
                score += self.transitions[previous_tag_id, tag]+word_scores[tag]
                previous_tag_id = tag
            scores[i] = score
        return scores

    """
    Sentence level log likelihood
    input: output of network (batch,seqlen,num_tags)
    real_tags: (batch_size,sequence_length) real tags to compare the results to
    """
    def sentence_level_log_likelihood(self, network_out, real_tags):
        sequence_length = network_out.shape[1]

        S = self.score_of_sequence(network_out, real_tags)
        scores = Variable(self.float_type(len(S)))

        for batch_num, batch_score in enumerate(S):
            backward = self.float_type(sequence_length, self.number_tags).zero_() - float('inf')
            backward = Variable(backward)

            backward[sequence_length-1, :] = network_out[batch_num, sequence_length-1, :]

            for pos in range(sequence_length-2, -1, -1):
                for state in range(self.number_tags):
                    backward[pos, state] = \
                        self.logops.logsumexp(backward[pos+1, :] + self.transitions[:, state]) +\
                        network_out[batch_num, pos+1, state]

            log_likelihood = self.logops.logsumexp(backward[0, :] + self.initial_transitions) +\
                network_out[batch_num, 0, :]
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
    """
        words: (seq_len,window_size,batch_size)
        tags: (seq_len,batch_size)
    """
    def get_train_tensors(self, words, tags=[],  word2id=None, tag2id=None, suffix2id=None):
        assert len(words[0]) == 1+self.window_size*2
        assert len(words[0][0]) == self.batch_size
        assert len(words) == len(tags)

        word_tensor = torch.LongTensor(len(words), self.batch_size, 1+self.window_size*2)
        tag_tensor = torch.LongTensor(self.batch_size, len(words))
        cap_tensor = torch.LongTensor(len(words), self.batch_size, 1+self.window_size*2)
        suf_tensor = torch.LongTensor(len(words), self.batch_size, 1+self.window_size*2)
        for n_seq, window in enumerate(words):
            for pos, word_list in enumerate(window):
                for batch, word in enumerate(word_list):
                    word_tensor[n_seq][batch][pos] = word2id(word)
                    if self.include_cap:
                        cap_tensor[n_seq][batch][pos] = capitalization_analysis(word)
                        suf_tensor[n_seq][batch][pos] = suffix2id(word)
        if tag2id is not None:
            for n_seq, window_tags in enumerate(tags):
                for batch, tag in enumerate(window_tags):
                    tag_tensor[batch][n_seq] = tag2id(tag)

        if self.gpu and torch.cuda.is_available():
            return word_tensor.cuda(), tag_tensor.cuda(), cap_tensor.cuda(), suf_tensor.cuda()
        else:
            return word_tensor, tag_tensor, cap_tensor, suf_tensor
