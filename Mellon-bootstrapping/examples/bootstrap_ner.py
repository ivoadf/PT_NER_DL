from __future__ import print_function
from mellon_bootstrap import MellonBoot
from dictionary_annotate import annotate
import argparse
import re
import os
import random

random.seed()

def get_new_names(initial_list, current_list):
    return list(set(current_list)-set(initial_list))


parser = argparse.ArgumentParser(description='Mellon Bootstrap learning to detect People')
parser.add_argument("input_filename", help="relative path to raw text data")
parser.add_argument("initial_name_list", help="relative path to initial_name_list")
parser.add_argument("test_filename", help="relative path to test dataset")
parser.add_argument("-mt", "--manual_test_filename", help="relative path to manually annotated news test dataset")
parser.add_argument("-e", "--trained_emb_filename", help="relative path to pretrained embeddings")
parser.add_argument("output_dir", help="relative path to output directory")
parser.add_argument("-bs", "--batch_size", help="Batch size", type=int, default=1)
parser.add_argument("-it", "--max_iterations", help="Max number of bootstrapping iterations", type=int, default=20)
args = parser.parse_args()

os.mkdir(args.output_dir)

initial_name_list = open(args.initial_name_list, "r").read().splitlines()
intermediate_filename = "data/{}.mellon.parsed".format(os.path.basename(args.input_filename))
annotated_filename = annotate(args.input_filename, initial_name_list, intermediate_filename)

# stats file
stats_file = open(os.path.join(args.output_dir, "total_stats"), "w")
stats_file.write("Iteration,F1,Precision,Recall,NewNames,SmallF1,SmallPrecision,SmallRecall\n")

architecture = MellonBoot(annotated_filename, args.test_filename, args.trained_emb_filename, batch_size=args.batch_size)
iteration_counter = 0
prev_f1_score = 0
f1_score = 0

while iteration_counter < args.max_iterations:  # f1_score >= prev_f1_score and iteration_counter < 20:
    iteration_counter += 1
    print("\n\nITERATION {}".format(iteration_counter))
    # reload train dataset
    architecture.set_train_dataset(annotated_filename)
    architecture.train()
    prev_f1_score = f1_score
    acc, precision, recall, f1_score = architecture.test()
    print("F1 test score: {}".format(f1_score))
    name_list = architecture.name_list()
    new_names = get_new_names(initial_name_list, name_list)
    new_names_len = len(new_names)
    print("Raw names detected {}".format(len(new_names)))
    # single word names are only added if model precision is above threshold
    if f1_score < 50:
        new_names = [name for name in new_names if len(name.split(' ')) >= 2]
        print("Excluding single word names. Exluded {} names".format(new_names_len-len(new_names)))

    new_names_len = len(new_names)
    # remove names that include non alphabetic chars
    tmp_new_names = []
    excluded_names = []
    for name in new_names:
        if re.search(r'[^\w\s]', name.decode('utf-8'), flags=re.UNICODE) is None:
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
        #break
    initial_name_list += new_names
    print("{} new names added to the name list.".format(len(new_names)))
    # re annotate train file
    annotated_filename = annotate(args.input_filename, initial_name_list, intermediate_filename)

    # get performance on small news dataset
    if args.manual_test_filename is not None:
        small_acc, small_precision, small_recall, small_f1_score = architecture.test_file(args.manual_test_filename)
        print("Small F1 test score: {}".format(small_f1_score))
        # log to stats file
        stats_file.write("{},{},{},{},{},{},{},{}\n".format(iteration_counter, f1_score, precision,
                                                            recall, len(new_names), small_f1_score,
                                                            small_precision, small_recall))

    # save model and test values to output folder
    architecture.save_model(args.output_dir, iteration_counter)
    with open(os.path.join(args.output_dir, "iteration_{}".format(iteration_counter)+'.test'), "w") as f:
        f.write("Acc: {} Prec: {} Recall: {} F1 test score: {}".format(acc, precision, recall, f1_score))
        f.write("\nNew Names added:\n")
        f.write("\n".join(new_names))
