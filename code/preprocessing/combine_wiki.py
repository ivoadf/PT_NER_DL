import argparse
import re
import os
import nltk


def text_to_conll(text):
    result = ''
    sentence_list = nltk.tokenize.sent_tokenize(text, "portuguese")
    for i, s in enumerate(sentence_list):
        word_list = nltk.word_tokenize(s, "portuguese")
        result += ''.join([w+'\n' for w in word_list])
        result += '\n'
    return result

"""
    Combine WikiExtractor outputs and remove lines with parsing errors
    https://github.com/attardi/wikiextractor
"""

parser = argparse.ArgumentParser(
    description='Combine WikiExtractor outputs and remove lines with parsing errors')
parser.add_argument("dir_name", help="relative path to WikiExtractor output")
parser.add_argument("output_filename", help="relative path to output file")
args = parser.parse_args()

output_file = open(args.output_filename, 'w')

subdir_list = os.listdir(args.dir_name)
path_list = []
for subdir in subdir_list:
    filename_list = os.listdir(os.path.join(args.dir_name, subdir))
    path_list += [os.path.join(args.dir_name, subdir, filename) for filename in filename_list]

total_files = len(path_list)
counter = 1

for path in path_list:
    print("Processing {:10} / {}".format(counter, total_files))
    counter += 1

    f = open(path, 'r')
    data = f.read()
    # remove parsing errors and xml annotation
    data = re.sub(r'.*(\[\[|\]\]|<|>).*', '', data)
    # clear extra newlines
    data = re.sub(r'^\n+', '\n', data, flags=re.MULTILINE)
    data = text_to_conll(data)
    output_file.write(data)
    f.close()

output_file.close()
