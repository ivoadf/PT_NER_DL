import time
import math
from torch.utils.data import DataLoader
from architectures.loaders.conll_dataset import ConllWindowFromFile

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


"""
Return words that belong to entities
States:
    1 - Only O tags before
    2 - S tag
    3 - B TAG
"""


def get_entities(words, tags):
    entities = []
    entity = []
    state = 1
    for i, tag in enumerate(tags):
        # transitions from state 1
        if state == 1:
            if tag[0] == 'S':
                state = 2
            elif tag[0] == 'B':
                state = 3
        # transition from state 2
        if state == 2:
            entities.append(words[i])
            state = 1
        # transitions from state 3
        elif state == 3:
            entity.append(words[i])
            if tag[0] == 'E':
                entities.append(' '.join(entity))
                entity = []
                state = 1
                continue
    return entities


"""
    Transform IOBES notation to IOB2
    States:
        1 - Only O tags before
        2 - S tag
        3 - B TAG
"""


def iobes_to_iob(iobes_tag_list):
    iob_tag_list = []
    state = 1
    for tag in iobes_tag_list:
        # transitions from state 1
        if state == 1:
            if tag[0] == 'S':
                state = 2
            elif tag[0] == 'B':
                state = 3
            else:
                iob_tag_list.append('O')
                continue
        # transition from state 2
        if state == 2:
            iob_tag_list.append('B'+tag[1:])
            state = 1
        # transitions from state 3
        elif state == 3:
            pre = tag[0]
            if tag[0] == 'E':
                pre = 'I'
            iob_tag_list.append(pre+tag[1:])
            if tag[0] == 'E':
                state == 1
                continue
    assert len(iob_tag_list) == len(iobes_tag_list)
    return iob_tag_list
