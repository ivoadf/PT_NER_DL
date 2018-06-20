from architectures.architecture import Architecture
from architectures.networks.collobert import NN
from architectures.loaders.batch_sampler import BatchSampler
from architectures.loaders.conll_dataset import ConllWindowFromFile
from architectures.loaders.embeddings_loader import EmbeddingsLoader
from architectures.utils.helper import get_entities, iobes_to_iob
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import os

NUMBER_TAGS = 4
LEARNING_RATE = 0.01
EPOCHS = 3
HIDDEN_SIZE = 300
EVAL_PATH = './architectures/evaluation/'
EVAL_SCRIPT = os.path.join(EVAL_PATH, 'conlleval.pl')
WINDOW_SIZE = 2


class Collobert(Architecture):
    def __init__(self, batch_size=1, silent=True, gpu=False):
        self.batch_size = batch_size
        self.gpu = gpu
        self.silent = silent
        self.network = None
        self.train_loader = None
        self.test_loader = None
        self.embeddings = None
        self.embedding_dim = 10
        self.vocab_size = 10
        self.word_to_index = None
        self.tag_to_index = None
        self.suffix_to_index = None
        self.index_to_tag = None
        self.num_suffix = None
        if self.gpu and torch.cuda.is_available():
            self.v_type = torch.cuda.FloatTensor
        else:
            self.v_type = torch.FloatTensor

    def train(self):
        assert self.train_loader is not None and self.word_to_index is not None and\
         self.tag_to_index is not None

        network = NN(vocab_size=self.vocab_size, number_tags=NUMBER_TAGS,
                     number_suffix=self.num_suffix, embedding_dim=self.embedding_dim,
                     hidden_size=HIDDEN_SIZE, include_cap=True, gpu=self.gpu,
                     batch_size=self.batch_size)

        if self.embeddings is not None:
            assert self.vocab_size == self.embeddings.get_dimensions()[0] and\
                self.embedding_dim == self.embeddings.get_dimensions()[1]
            network.set_embeddings_weights(self.embeddings.get_weights())

        model_parameters = [p for p in network.parameters() if p.requires_grad]
        params = sum([p.numel() for p in model_parameters])
        print("Model with {} trainable parameters".format(params))

        optimizer = torch.optim.SGD(model_parameters, lr=LEARNING_RATE)

        if self.gpu and torch.cuda.is_available():
            network.cuda()

        network.train()
        for epoch in range(EPOCHS):
            for it, (words, tags) in enumerate(self.train_loader):
                word_tensor, tag_tensor, cap_tensor, suf_tensor = \
                    network.get_train_tensors(words,
                                              tags,
                                              word2id=self.word_to_index,
                                              tag2id=self.tag_to_index,
                                              suffix2id=self.suffix_to_index)
                # reset gradients, else they accumulate
                optimizer.zero_grad()

                acc_output = Variable(self.v_type(self.batch_size, len(word_tensor),
                                                  NUMBER_TAGS))
                # loop sequence
                for seq_n in range(len(word_tensor)):

                    output = network(Variable(word_tensor[seq_n]),
                                   Variable(cap_tensor[seq_n]),
                                   Variable(suf_tensor[seq_n]))
                    acc_output[:, seq_n, :] = output

                score = network.sentence_level_log_likelihood(acc_output, tag_tensor)

                score.backward()
                optimizer.step()

                print('', end='\r')
                print("Epoch {:5} Iteration {:5} Score {:7.5f}".format(epoch,
                                                                       it,
                                                                       score.data[0]),
                      end='')
            print('')

        self.network = network

    def test(self):
        predictions = []
        self.network.eval()
        for words, tags in self.test_loader:
            word_tensor, tag_tensor, cap_tensor, suf_tensor =\
                    self.network.get_train_tensors(words,
                                              tags,
                                              word2id=self.word_to_index,
                                              suffix2id=self.suffix_to_index)
            acc_output = Variable(self.v_type(self.batch_size, len(word_tensor),
                                         NUMBER_TAGS))
            # loop sequence
            for seq_n in range(len(word_tensor)):

                output = self.network(Variable(word_tensor[seq_n]),
                               Variable(cap_tensor[seq_n]),
                               Variable(suf_tensor[seq_n]))
                acc_output[:, seq_n, :] = output

            words_and_true_tags = zip(words, tags)
            for batch in range(acc_output.shape[0]):
                _, tag_list = self.network.viterbi_decode(acc_output[batch])
                match = [(w[WINDOW_SIZE][batch], t[batch]) for w, t in words_and_true_tags]
                tag_list = [self.index_to_tag(t) for t in tag_list]
                iob_tag_list = iobes_to_iob(tag_list)
                predictions += ["{} {} {}".format(w, true_t, pred_t) for
                                (w, true_t), pred_t in zip(match, iob_tag_list)]
                predictions.append("")
        # Write predictions to disk and run CoNLL script externally
        output_path = os.path.join(EVAL_PATH, "eval.output")
        scores_path = os.path.join(EVAL_PATH, "eval.scores")
        with open(output_path, 'w') as f:
            f.write("\n".join(predictions))
        os.system("%s < %s > %s" % (EVAL_SCRIPT, output_path, scores_path))

        # CoNLL evaluation results
        eval_lines = [l.rstrip() for l in open(scores_path, 'r')]
        return_string = ""
        for line in eval_lines:
            return_string += line+'\n'
        return float(eval_lines[1].strip().split()[-1]), return_string

    def name_list(self):
        total_name_dict = {}
        self.network.eval()
        for it, (words, tags) in enumerate(self.train_loader):
            word_tensor, tag_tensor, cap_tensor, suf_tensor =\
                    self.network.get_train_tensors(words,
                                              tags,
                                              word2id=self.word_to_index,
                                              tag2id=self.tag_to_index,
                                              suffix2id=self.suffix_to_index)
            acc_output = Variable(self.v_type(self.batch_size, len(word_tensor),
                                         NUMBER_TAGS))
            # loop sequence
            for seq_n in range(len(word_tensor)):

                output = self.network(Variable(word_tensor[seq_n]),
                               Variable(cap_tensor[seq_n]),
                               Variable(suf_tensor[seq_n]))
                acc_output[:, seq_n, :] = output

            words_and_true_tags = zip(words, tags)
            for batch in range(acc_output.shape[0]):
                _, tag_list = self.network.viterbi_decode(acc_output[batch])
                batch_words = [w[WINDOW_SIZE][batch] for w in words]
                tag_list = [self.index_to_tag(t) for t in tag_list]
                names = get_entities(batch_words, tag_list)
                for name in names:
                    if name in total_name_dict:
                        total_name_dict[name] += 1
                    else:
                        total_name_dict[name] = 1
        return set([name for name, num_occ in total_name_dict.items() if num_occ >= 4])

    def set_test_dataset(self, test_filename):
        using_embeddings = self.embeddings is None
        test_dataset = ConllWindowFromFile(test_filename,
                                           window_size=WINDOW_SIZE,
                                           vocab_size=self.vocab_size,
                                           compute_dicts=using_embeddings)
        test_batch_sampler = BatchSampler(test_dataset, self.batch_size, enforce=True)
        self.test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler)

        print("Total testing examples: {} | Total testing batches: {} | Batch Size: {}".format(
            len(test_dataset), len(test_batch_sampler), self.batch_size
        ))

    def set_train_dataset(self, train_filename):
        using_embeddings = self.embeddings is None
        train_dataset = ConllWindowFromFile(train_filename,
                                            window_size=WINDOW_SIZE,
                                            vocab_size=self.vocab_size,
                                            compute_dicts=using_embeddings)
        train_batch_sampler = BatchSampler(train_dataset, self.batch_size, enforce=True)
        self.train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
        self.tag_to_index = train_dataset.tag_to_index
        if self.embeddings is None:
            self.word_to_index = train_dataset.word_to_index
        self.index_to_tag = train_dataset.index_to_tag
        self.suffix_to_index = train_dataset.suffix_to_index
        self.num_suffix = train_dataset.get_num_suffix()
        print("Total training examples: {} | Total training batches: {} | Batch Size: {}".format(
            len(train_dataset), len(train_batch_sampler), self.batch_size
        ))

    def set_embeddings(self, embeddings_filename):
        self.embeddings = EmbeddingsLoader(embeddings_filename)
        self.vocab_size, self.embedding_dim = self.embeddings.get_dimensions()
        self.word_to_index = self.embeddings.word_to_index

    def save_model(self, save_path, bootstrapping_iteration):
        save_name = "COLLOBERT_iteration_{}_epochs_{}_LR_{}_vocabsize_{}_embdim_{}".format(
            EPOCHS,
            bootstrapping_iteration,
            LEARNING_RATE,
            self.vocab_size,
            self.embedding_dim)
        torch.save(self.network.state_dict(), os.path.join(save_path, save_name+'.model'))
