import time
import math
import os
from torch.utils.data import DataLoader
from loaders.conll_dataset import ConllWindowFromFile

def print_tag_dict(train_tags,predicted_train_tags,test_tags,predicted_test_tags):
    for key in train_tags.keys():
        if key not in predicted_train_tags:
            predicted_train_tags[key] = 0
        if key not in predicted_test_tags:
            predicted_test_tags[key] = 0
        if key not in test_tags:
            test_tags[key] = 0
    train_tags_s = sorted(train_tags.items(), key = lambda x: x[0])
    predicted_train_tags_s = sorted(predicted_train_tags.items(), key = lambda x: x[0])
    test_tags_s = sorted(test_tags.items(), key = lambda x: x[0])
    predicted_test_tags_s = sorted(predicted_test_tags.items(), key = lambda x: x[0])
    print("\n"+"-"*91)
    print("| {:^15} | {:^15} | {:^15} | {:^15} | {:^15} |".format("TAG","TRAIN","P-TRAIN","TEST","P-TEST"))
    for (k1,k2,k3,k4) in zip(train_tags_s,predicted_train_tags_s,test_tags_s,predicted_test_tags_s):
        print("| {:15} | {:15} | {:15} | {:15} | {:15} |".format(k1[0],k1[1],k2[1],k3[1],k4[1]))
    print("-"*91)
    train_tot = sum([v for k,v in train_tags_s])
    predicted_train_tot = sum([v for k,v in predicted_train_tags_s])
    test_tot = sum([v for k,v in test_tags_s])
    predicted_test_tot = sum([v for k,v in predicted_test_tags_s])
    print("| {:15} | {:15} | {:15} | {:15} | {:15} |".format('TOTAL',train_tot,predicted_train_tot,
    test_tot,predicted_test_tot))
    print("-"*91)

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{}m:{:.2f}s'.format(m, s)
"""
5 possibilities: all lowercased, first uppercased, all uppercased, contains an uppercased letter, and all other cases
"""
def capitalization_analysis(word):
    if word.islower():
        return 0
    elif word.isupper():
        return 1
    elif word[0].isupper() and word[1:].islower():
        return 2
    elif len([c for c in word if c.isupper()]) > 0:
        return 3
    else:
        return 4


def evaluate(output_file):
    score_file = "tmp/score_collobert"
    os.system("./conll03eval.v2 < %s > %s" % (output_file, score_file))
    print(open(score_file, "r").read())
    with open(score_file, 'r') as fin:
        fin.readline()
        line = fin.readline()
        fields = line.split(";")
        acc = float(fields[0].split(":")[1].strip()[:-1])
        precision = float(fields[1].split(":")[1].strip()[:-1])
        recall = float(fields[2].split(":")[1].strip()[:-1])
        f1 = float(fields[3].split(":")[1].strip())
    return acc, precision, recall, f1
