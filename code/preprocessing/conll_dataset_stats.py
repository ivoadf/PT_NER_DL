import argparse
import math
parser = argparse.ArgumentParser(description='Display stats of conll dataset')
parser.add_argument("input", help="relative path to conll dataset")
args = parser.parse_args()

f = open(args.input, "r")
stats = {
    'num_sentences': 0,
    'num_words': 0,
    'max_sentence_length': math.inf,
    'min_sentence_length': -1,
    'average_sentence_length': -1,
    'number_annotations': 0
}
sentences = {}
sentence_length_counter = 0
in_entity = False
for line in f:
    if line == "-DOCSTART- -X- O O":
        continue
    if line == '\n':
        if sentence_length_counter in sentences:
            sentences[sentence_length_counter] += 1
        else:
            sentences[sentence_length_counter] = 1
        stats['num_words'] += sentence_length_counter
        stats['num_sentences'] += 1
        sentence_length_counter = 0
    else:
        sentence_length_counter += 1
        tag = line.split()[1][0]
        if tag == "B":
            stats['number_annotations'] += 1

if sentence_length_counter != 0:
    if sentence_length_counter in sentences:
        sentences[sentence_length_counter] += 1
    else:
        sentences[sentence_length_counter] = 1
    stats['num_words'] += sentence_length_counter
    stats['num_sentences'] += 1

f.close()

stats['max_sentence_length'] = max(sentences.items(), key=lambda x:  x[0])[0]
stats['min_sentence_length'] = min(sentences.items(), key=lambda x:  x[0])[0]
stats['average_sentence_length'] = sum([k*v for k, v in sentences.items()])\
                        / sum(v for _, v in sentences.items())

for key, value in stats.items():
    print("{:30} | {:10}".format(key, value))
print("\n\n")

total_sentences = sum(v for _, v in sentences.items())
sentence_counter = 0
percentage_counter = 0.1
for key, value in sentences.items():
    sentence_counter += value
    if sentence_counter/total_sentences > percentage_counter:
        print("{:.1f} percent of all sentences are under size {:8}".format(percentage_counter, key))
        percentage_counter += 0.1
