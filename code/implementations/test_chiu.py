from models.chiu import Network
import torch
from torch.autograd import Variable

lstm = Network(22,
               3,
               retrain_embeddings=True,
               embedding_dim=25,
               hidden_size=25,
               dropout=0)

optimizer = torch.optim.SGD(lstm.parameters(), lr=0.01)

# data prep
sentences = ["O João Pedro foi para a escola hoje de manhã .", "A Maria comeu a sopa toda .",
             "Ontem o Alfredo Matias fugiu da escola ."]
tags = ["O B-PER I-PER O O O O O O O O", "O B-PER O O O O O", "O O B-PER I-PER O O O O"]

tag2id = {"O": 0, "B-PER": 1, "I-PER": 2}
id2tag = {0: "O", 1: "B-PER", 2: "I-PER"}
word2id = {}
id2word = {}
id_count = 0
for sentence in sentences:
    words = sentence.split()
    for w in words:
        if w not in word2id:
            word2id[w] = id_count
            id2word[id_count] = w
            id_count += 1
inputs = []
targets = []

for (sentence, tag_sentence) in zip(sentences, tags):
    words = sentence.split()
    word_ids = [word2id[w] for w in words]
    tag_list = tag_sentence.split()
    tag_ids = [tag2id[tag] for tag in tag_list]

    inputs.append(Variable(torch.LongTensor([word_ids])))
    targets.append(torch.LongTensor([tag_ids]))

# train model
for i in range(100):
    for (input, target) in zip(inputs,targets):
        o = lstm(input)
        optimizer.zero_grad()
        _, tag_list = lstm.viterbi_decode(o[0])
        score = lstm.sentence_level_log_likelihood(o, target)
        print(score.data[0], end='\r')
        score.backward()
        optimizer.step()
        if i == 99:
            print('\n')
            print([id2word[w] for w in input.data[0]])
            print([id2tag[tag] for tag in tag_list.tolist()])
