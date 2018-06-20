import argparse
"""
    conll2003 has the following categories:
        PER, MISC, ORG, LOC
    HAREM selective scenario will have:
        Person, Organization, Location, Abstraction
"""

selective_set = {'ORGANIZACAO', 'ABSTRACCAO', 'LOCAL', 'PESSOA'}
tag_mapping = {'ORGANIZACAO': 'ORG', 'ABSTRACCAO': 'MISC',
               'LOCAL': 'LOC', 'PESSOA': 'PER'}

parser = argparse.ArgumentParser(
    description='Transform conll harem dataset into selective harem')
parser.add_argument("input", help="relative path to input")
parser.add_argument("output", help="relative path to output file")
args = parser.parse_args()

f = open(args.input, 'r')
out = open(args.output, 'w')

for line in f:
    s = line.split()
    if len(s) > 1:
        word = s[0]
        tag = s[1]
        if tag == 'O':
            out.write(line)
        elif tag[2:] in selective_set:
            new_tag = "{}{}".format(tag[:2], tag_mapping[tag[2:]])
            out.write("{} {}\n".format(word, new_tag))
        else:
            out.write("{} O\n".format(word))
    else:
        out.write('\n')

f.close()
out.close()
