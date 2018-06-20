from architectures.architecture import Architecture
from architectures.networks.chiu import Network
from architectures.loaders.batch_sampler import BatchSampler
from architectures.loaders.conll_dataset import ConllSequenceFromFile
from architectures.loaders.embeddings_loader import EmbeddingsLoader
from architectures.utils.helper import get_entities, iobes_to_iob
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import os

NUMBER_TAGS = 4
LEARNING_RATE = 0.015
EPOCHS = 20
HIDDEN_SIZE = 200
DROPOUT = 0.6
EVAL_PATH = './architectures/evaluation/'
EVAL_SCRIPT = os.path.join(EVAL_PATH, 'conlleval.pl')


class Chiu(Architecture):

    def __init__(self, batch_size=1, gpu=False):
        self.batch_size = batch_size
        self.gpu = gpu
        self.network = None
        self.train_loader = None
        self.test_loader = None
        self.embeddings = None
        self.embedding_dim = 10
        self.vocab_size = 10
        self.word_to_index = None
        self.tag_to_index = None
        self.index_to_tag = None

    def train(self):
        assert self.train_loader is not None and self.word_to_index is not None and\
         self.tag_to_index is not None

        network = Network(vocab_size=self.vocab_size, number_tags=NUMBER_TAGS,
                          embedding_dim=self.embedding_dim, hidden_size=HIDDEN_SIZE,
                          dropout=DROPOUT, include_cap=True, gpu=self.gpu,
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
                word_tensor, tag_tensor, cap_tensor = \
                    network.get_train_tensors(words,
                                              tags,
                                              word2id=self.word_to_index,
                                              tag2id=self.tag_to_index)
                optimizer.zero_grad()
                s = network(Variable(word_tensor), Variable(cap_tensor))
                score = network.sentence_level_log_likelihood(s, tag_tensor)
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
            word_tensor, tag_tensor, cap_tensor =\
                self.network.get_train_tensors(words,
                                               tags,
                                               word2id=self.word_to_index,
                                               tag2id=self.tag_to_index)
            s = self.network(Variable(word_tensor), Variable(cap_tensor))
            words_and_true_tags = zip(words, tags)
            for batch in range(s.shape[0]):
                _, tag_list = self.network.viterbi_decode(s[batch])
                match = [(w[batch], t[batch]) for w, t in words_and_true_tags]
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

    """
        Only assign new name if it accurs at least 4 times
    """
    def name_list(self):
        total_name_dict = {}
        self.network.eval()
        for it, (words, tags) in enumerate(self.train_loader):
            word_tensor, tag_tensor, cap_tensor =\
                self.network.get_train_tensors(words,
                                               tags,
                                               word2id=self.word_to_index,
                                               tag2id=self.tag_to_index)
            s = self.network(Variable(word_tensor), Variable(cap_tensor))
            for batch in range(s.shape[0]):
                _, tag_list = self.network.viterbi_decode(s[batch])
                batch_words = [w[batch] for w in words]
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
        test_dataset = ConllSequenceFromFile(test_filename,
                                             vocab_size=self.vocab_size,
                                             compute_dicts=using_embeddings)
        test_batch_sampler = BatchSampler(test_dataset, self.batch_size, enforce=True)
        self.test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler)

        print("Total testing examples: {} | Total testing batches: {} | Batch Size: {}".format(
            len(test_dataset), len(test_batch_sampler), self.batch_size
        ))

    def set_train_dataset(self, train_filename):
        using_embeddings = self.embeddings is None
        train_dataset = ConllSequenceFromFile(train_filename,
                                              vocab_size=self.vocab_size,
                                              compute_dicts=using_embeddings)
        train_batch_sampler = BatchSampler(train_dataset, self.batch_size, enforce=True)
        self.train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
        self.tag_to_index = train_dataset.tag_to_index
        if self.embeddings is None:
            self.word_to_index = train_dataset.word_to_index
        self.index_to_tag = train_dataset.index_to_tag

        print("Total training examples: {} | Total training batches: {} | Batch Size: {}".format(
            len(train_dataset), len(train_batch_sampler), self.batch_size
        ))

    def set_embeddings(self, embeddings_filename):
        self.embeddings = EmbeddingsLoader(embeddings_filename)
        self.vocab_size, self.embedding_dim = self.embeddings.get_dimensions()
        self.word_to_index = self.embeddings.word_to_index

    def save_model(self, save_path, bootstrapping_iteration):
        save_name = "CHIU_iteration_{}_epochs_{}_LR_{}_vocabsize_{}_embdim_{}".format(
            EPOCHS,
            bootstrapping_iteration,
            LEARNING_RATE,
            self.vocab_size,
            self.embedding_dim)
        torch.save(self.network.state_dict(), os.path.join(save_path, save_name+'.model'))
