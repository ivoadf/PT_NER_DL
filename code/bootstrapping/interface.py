import argparse
import re
import os
from dictionary_annotate import annotate
from architectures.chiu import Chiu
from architectures.loaders.embeddings_loader import EmbeddingsLoader
from architectures.collobert import Collobert
"""
load raw dataset
load list of names N
dictionary style annotate with list N
load annotated dataset
train model
apply model and obtain list of new names identified L
dictionary style annotate with list N+L
repeat until stable
"""


def get_new_names(initial_list, current_list):
    return list(set(current_list)-set(initial_list))


parser = argparse.ArgumentParser(description='Bootstrap learning to detect People')
parser.add_argument("input_filename", help="relative path to raw text data")
parser.add_argument("initial_name_list", help="relative path to initial_name_list")
parser.add_argument("test_filename", help="relative path to test dataset")
parser.add_argument("-e", "--trained_emb_filename", help="relative path to pretrained embeddings")
parser.add_argument("architecture", help="DL architecture to train",
                    choices=['chiu', 'collobert'])
parser.add_argument("output_dir", help="relative path to output directory")
parser.add_argument("-G", "--gpu", help="Use GPU if available", action="store_true")
parser.add_argument("-bs", "--batch_size", help="Batch size", type=int, default=1)
args = parser.parse_args()

if args.architecture == 'chiu':
    architecture = Chiu(batch_size=args.batch_size, gpu=args.gpu)
elif args.architecture == 'collobert':
    architecture = Collobert(batch_size=args.batch_size, gpu=args.gpu)

os.mkdir(args.output_dir)

initial_name_list = open(args.initial_name_list, "r").read().splitlines()

architecture.set_test_dataset(args.test_filename)
if args.trained_emb_filename is not None:
    architecture.set_embeddings(args.trained_emb_filename)

intermediate_filename = "i_data/{}.{}.parsed".format(os.path.basename(args.input_filename),
                                                     args.architecture)

iteration_counter = 0
prev_f1_score = 0
f1_score = 0

MAX_ITERATIONS = 20
while iteration_counter < MAX_ITERATIONS:  # f1_score >= prev_f1_score and iteration_counter < 20:
    iteration_counter += 1
    annotated_filename = annotate(args.input_filename, initial_name_list, intermediate_filename)
    architecture.set_train_dataset(annotated_filename)
    architecture.train()
    prev_f1_score = f1_score
    f1_score, test_log = architecture.test()
    print(test_log)
    name_list = architecture.name_list()
    new_names = get_new_names(initial_name_list, name_list)
    new_names_len = len(new_names)
    # single word names are only added if model precision is above threshold
    if f1_score < 50:
        new_names = [name for name in new_names if len(name.split(' ')) >= 2]
        print("Excluding single word names. Exluded {} names".format(new_names_len-len(new_names)))

    new_names_len = len(new_names)
    # remove names that include non alphabetic chars
    tmp_new_names = []
    excluded_names = []
    for name in new_names:
        if re.search(r'[^\w\s]', name) is None:
            tmp_new_names.append(name)
        else:
            excluded_names.append(name)
    new_names = tmp_new_names
    print("Exluding names with non alphabetic characters. Exluded {} names.".format(
        new_names_len-len(new_names)))
    if len(excluded_names) < 10:
        max_print = len(excluded_names)
    else:
        max_print = 10
    print("Excluded new names include:", excluded_names[:max_print])

    if len(new_names) == 0:
        print("No new names found. Aborting.")
        break
    initial_name_list += new_names
    print("{} new names added to the name list.".format(len(new_names)))
    # save model and test values to output folder
    architecture.save_model(args.output_dir, iteration_counter)
    with open(os.path.join(args.output_dir, "iteration_{}".format(iteration_counter)+'.test'), "w") as f:
        f.write(test_log)
        f.write("\nNew Names added:\n")
        f.write("\n".join(new_names))
