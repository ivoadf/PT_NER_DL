import json
import argparse
import os
import sys
import nltk
from dateutil.parser import parse
import matplotlib.pyplot as plt

def text_to_conll(text):
    result = ''
    sentence_list = nltk.tokenize.sent_tokenize(text, "portuguese")
    for i, s in enumerate(sentence_list):
        word_list = nltk.word_tokenize(s, "portuguese")
        result += ''.join([w+'\n' for w in word_list])
        result += '\n'
    return result


parser = argparse.ArgumentParser(
    description='Parse json news data.')
parser.add_argument("dir_name", help="relative path to News json files dir")
parser.add_argument("output_filename", help="relative path to output file")
parser.add_argument("-s", "--stats", help="Just run stats", action="store_true")
args = parser.parse_args()

output_file_conll = open(args.output_filename+".conll", 'w')
output_file_raw = open(args.output_filename+".raw", 'w')

filename_list = os.listdir(args.dir_name)
path_list = [os.path.join(args.dir_name, filename) for filename in filename_list]

total_files = len(path_list)
counter = 0
news_counter = 0
dates = {}

for filename in path_list:
    f = open(filename, "r")
    json_obj = json.load(f)
    f.close()
    docs = json_obj['response']['docs']
    for doc in docs:
        if doc['Lang'] == 'pt':
            date = parse(doc['Pubdate'])
            datestring = "{}/{}".format(date.month, date.year)
            if datestring in dates:
                dates[datestring] += 1
            else:
                dates[datestring] = 1
            news_counter += 1
            if not args.stats:
                if 'Title' in doc.keys():
                    data = '\n'.join([doc['Title'], doc['Body']])
                else:
                    data = doc['Body']
                data_conll = text_to_conll(data)
                output_file_conll.write(data_conll)
                output_file_raw.write(data+'\n')
    counter += 1
    print("Processed {} of {} documents. Total news processed {}.".format(
        counter, total_files, news_counter), end='\r')

print('\n')
output_file_conll.close()
output_file_raw.close()


def d_sort(x):
    s = x[0].split("/")
    if int(s[0]) < 10:
        s[0] = "0"+s[0]
    return int(s[1]+s[0])


dates = sorted(dates.items(), key=d_sort)
vals = [v for _, v in dates]
keys = [k for k, _ in dates]
plt.switch_backend('agg')  # preventing error if no display manager running
plt.bar(range(len(dates)), vals, align='center')
plt.xticks(range(len(dates)), keys)
plt.ylabel("Number docs")
plt.xlabel("Publish date")
plt.savefig(os.path.join(args.output_filename+'.jpg'))
plt.close()
