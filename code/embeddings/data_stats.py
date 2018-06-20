import argparse
from os import path

parser = argparse.ArgumentParser(description='Analyze text file statistics')
parser.add_argument("file", help="relative path to file")
args = parser.parse_args()

with open(args.file,"r") as f, open(args.file+'.stats',"w") as out:
    file_size = path.getsize(args.file)
    total = 0
    for line in f:
        n_words = len(line.split(' '))
        total += n_words
    out.write("Total number of words: {}\n".format(total))
    out.write("Number of bytes: {}\n".format(file_size))
