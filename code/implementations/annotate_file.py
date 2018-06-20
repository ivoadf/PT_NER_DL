from nn import NN
from embeddings_loader import EmbeddingsLoader
from torch.autograd import Variable
from helper import loadDatasets
import argparse
import torch

parser = argparse.ArgumentParser(description='Script to annotate test dataset, output can be used on the Conll evaluation script.')
parser.add_argument("trained_model", help="relative path to trained model")
parser.add_argument("output_file", help="relative path to output file")
parser.add_argument("-d","--train_dataset", help="relative path to conll style train dataset")
parser.add_argument("-td","--test_dataset", help="relative path to conll style test dataset")
parser.add_argument("-bs","--batch_size", help="batch size",default=1, type=int)
parser.add_argument("-G","--gpu", help="Use GPU if available.", action="store_true", default=False)
parser.add_argument("-C","--capitalization", help="Include capitalization feature",action="store_true")
parser.add_argument("-A","--activation", help="Activation function to use",choices=['relu','hardtanh','tanh'],default='hardtanh')
parser.add_argument("-ep","--embeddings_path", help="relative path to pretrained embeddings")
parser.add_argument("-ws","--window_size", help="size of window",default=2, type=int)
parser.add_argument("-vs","--vocab_size", help="size of vocab",default=100000, type=int)
parser.add_argument("-es","--embedding_size", help="embedding dimensions",default=200, type=int)
args = parser.parse_args()

dataset,loader,test_dataset,test_loader = loadDatasets(args.train_dataset,
args.test_dataset,args.window_size,args.batch_size,shuffle=True,vocab_size=args.vocab_size,compute_dicts = args.embeddings_path is None)
print("Number of training examples / test examples: {} / {}".format(len(dataset),len(test_dataset)))

if args.embeddings_path is not None:
    embLoader = EmbeddingsLoader(args.embeddings_path)
    print("Embeddings loaded, vocab size: {} embedding dims: {}".format(embLoader.get_dimensions()[0],embLoader.get_dimensions()[1]))
    args.vocab_size,args.embedding_size = embLoader.get_dimensions()

model = NN(vocab_size=args.vocab_size,number_tags=dataset.get_num_tags(),
batch_size=args.batch_size,gpu=args.gpu,window_size=args.window_size,
include_cap=args.capitalization,number_suffix=dataset.get_num_suffix(),
embedding_dim=args.embedding_size,activation=args.activation)

model.load_state_dict(torch.load(args.trained_model))

if args.gpu and torch.cuda.is_available():
    model.cuda()

tagged_file = open(args.output_file,"w")

if args.embeddings_path is not None:
    word_to_index = embLoader.word_to_index
else:
    word_to_index = dataset.word_to_index

for i,(words_input,tags) in enumerate(test_loader):
    input_tensor,target_tensor,cap_tensor,suf_tensor = model.get_train_tensors(words_input,tags,
    word_to_index,dataset.tag_to_index,suffix2id=dataset.suffix_to_index)
    output = model(Variable(input_tensor),Variable(cap_tensor),Variable(suf_tensor))
    top_n, top_i = output.data.topk(1)
    for i,tag_i in enumerate(top_i):
        word = words_input[args.window_size][i]
        real_tag = dataset.index_to_tag(target_tensor[i])
        predicted_tag = dataset.index_to_tag(tag_i[0])
        tagged_file.write("{} {} {}\n".format(word,real_tag,predicted_tag))

tagged_file.close()
