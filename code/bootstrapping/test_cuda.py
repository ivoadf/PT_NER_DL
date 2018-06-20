import argparse
from architectures.loaders.batch_sampler import BatchSampler
from architectures.loaders.conll_dataset import ConllWindowFromFile
from architectures.networks.collobert import NN
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Test dataloading impact on CUDA')
parser.add_argument("data", help="relative path to conll style test data")
args = parser.parse_args()

BATCH_SIZE = 16
# without preloading
train_dataset = ConllWindowFromFile(args.data,
                                    window_size=2,
                                    vocab_size=100000,
                                    compute_dicts=True)

train_batch_sampler = BatchSampler(train_dataset, BATCH_SIZE, enforce=True)
train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)

network = NN(vocab_size=100000, number_tags=4,
             number_suffix=train_dataset.get_num_suffix(), embedding_dim=100,
             hidden_size=256, include_cap=True, gpu=True,
             batch_size=BATCH_SIZE)

for it, (words, tags) in enumerate(train_loader):
    word_tensor, tag_tensor, cap_tensor, suf_tensor = \
        network.get_train_tensors(words,
                                  tags,
                                  word2id=train_dataset.word_to_index,
                                  tag2id=train_dataset.tag_to_index,
                                  suffix2id=train_dataset.suffix_to_index)
    del word_tensor, tag_tensor, cap_tensor, suf_tensor
    if it % 100 == 0:
        print("Iteration {}".format(it))
