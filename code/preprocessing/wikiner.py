import argparse
from helper import is_valid_file
import re
import xml.etree.ElementTree as ET
from more_itertools import peekable

parser = argparse.ArgumentParser(description='Process WikiNER dataset into standard XML notation.')
parser.add_argument("filename", help="relative path to wikiner data file",type=lambda x: is_valid_file(parser, x))
parser.add_argument("output_filename",help="relative path to raw output file")
parser.add_argument("--style",choices=['xml','conll','raw'],help="Output file style",default="conll")
parser.add_argument("-s","--scheme",choices=['IOB1','IOB2','IOE1','IOE2','IOBES'],help="Annotation scheme to use for conll style",default="IOB2")
parser.add_argument("--only_people", help="Only annotate people", action="store_true")
args = parser.parse_args()

#XML builder
tree_builder = ET.TreeBuilder()
tree_builder.start('wikiner',{})

def subprocess_xml(matches,tree_builder):
    last_ner_tag = 'O'
    for m in matches:
        (word,pos,ner) = m.groups()
        #xml style
        if ner != 'O':
            ner = ner[2:]
        if last_ner_tag == ner:
            tree_builder.data(word+' ')
        else:
            if last_ner_tag == 'O':
                tree_builder.start('EM',{'CATEG':ner})
                tree_builder.data(word+' ')
            elif ner == 'O':
                tree_builder.end('EM')
                tree_builder.data(word+' ')
        last_ner_tag = ner
def subprocess_conll(matches,style):
    #TODO wikiner available data is inconsistent, does not follow correctly the IOB1 scheme
    l_matches = list(matches)
    number_tokens = len(l_matches)
    matches = peekable(l_matches)

    processed_line = ''
    last_ner_tag = 'O'
    is_sequential = False
    for _ in range(number_tokens):
        m = next(matches)
        try:
            peek = matches.peek()
            (_,_,peek_ner) = peek.groups()
        except StopIteration:
            peek_ner = 'O'
        (word,pos,ner) = m.groups()
        if args.only_people and ner[2:] != 'PER':
            ner = 'O'
        if style == 'IOB1':
            ner = ner #wikiner already on IOB1 scheme
        elif style == 'IOB2':
            if last_ner_tag != ner and last_ner_tag == 'O':
                ner = 'B'+ner[1:]
            last_ner_tag = ner
        elif style == 'IOE1':
            if ner[0] == 'B':
                is_sequential = True
            if ner != 'O':
                if peek_ner == 'O' and is_sequential:
                    ner = 'E'+ner[1:]
                    is_sequential = False
                else:
                    ner = 'I'+ner[1:]
        elif style == 'IOE2':
            if ner != 'O':
                if peek_ner == 'O':
                    ner = 'E'+ner[1:]
                else:
                    ner = 'I'+ner[1:]
        processed_line += '\t'.join([word, ner])+'\n'
    processed_line += '\n'
    return processed_line

def process_line(line):
    if args.style == "raw":
        if line == '\n':
            return ''
        raw_line = re.sub(r'\|[^\|]*\|[^\s]*','',line)
        raw_line += '\n'
        return raw_line
    else:
        #1st group - word, 2nd - POS tag, 3rd - NER tag
        matches = re.finditer(r'([^\|\s]*)\|([^\|]*)\|([^\s]*)',line)
        if args.style == "xml":
            subprocess_xml(matches,tree_builder)
        elif args.style == "conll":
            return subprocess_conll(matches,args.scheme)

with open(args.filename,'r') as f, open(args.output_filename,'w') as out:
    for line in f:
        pl = process_line(line)
        if args.style != 'xml':
            out.write(pl)
    if args.style == 'xml':
        tree_builder.end('wikiner')
        root = tree_builder.close()
        tree = ET.ElementTree(root)
        tree.write(args.output_filename,encoding='utf-8',xml_declaration=True)
