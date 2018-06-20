import argparse

parser = argparse.ArgumentParser(
    description='Add indexes to conll file')
parser.add_argument("input", help="relative path to News json files dir")
parser.add_argument("output", help="relative path to output file")
args = parser.parse_args()

f = open(args.input, "r")
out = open(args.output, "w")
counter = 1
for line in f:
    line_s = line.split(' ')
    if line == '\n':
        counter = 1
    else:
        line = "{} {} {} {} {}".format(counter, line_s[0], 'O', 'O', line_s[1])
        counter += 1
    out.write(line)

f.close()
out.close()
