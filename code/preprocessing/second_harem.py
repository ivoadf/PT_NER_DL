import argparse
from os import listdir,path
import re
import sys
from nltk import word_tokenize
import nltk
from more_itertools import peekable
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description='Process HAREM GC into CoNLL scheme')
parser.add_argument("input", help="relative path to harem xml file")
parser.add_argument("output", help="relative path to output file")
parser.add_argument("-s","--scheme",choices=['IOB1','IOB2','IOE1','IOE2','IOBES'],help="Annotation scheme to use for conll style",default="IOBES")
args = parser.parse_args()

#files
out = open(args.output,'w')
metadata_file = open(args.output+'.metadata','w')

tree = ET.parse(args.input)
root = tree.getroot()

#structures
metadata = {
'num_docs': 0,
'num_words': 0,
'total_annotations': 0
}

def write_to_file(f,word_list):
    string = ' '.join(word_list)
    f.write(string+'\n')

def process_entity(word_list,out,class_name):
    class_name = class_name.split('|')[0] #opt for the first class when there are alternatives
    # if class_name != 'PESSOA':  #only annotate people
    #     for word in word_list:
    #         write_to_file(out,[word,'O'])
    #     return
    if args.scheme == "IOBES":
        if len(word_list) == 1:
            write_to_file(out,[word_list[0],'S-'+class_name])
        else:
            for i,w in enumerate(word_list):
                if i == 0:
                    write_to_file(out,[w,'B-'+class_name])
                elif i == len(word_list)-1:
                    write_to_file(out,[w,'E-'+class_name])
                else:
                    write_to_file(out,[w,'I-'+class_name])
    elif args.scheme == "IOB2":
        write_to_file(out,[word_list[0],'B-'+class_name])
        for w in word_list[1:]:
            write_to_file(out,[w,'I-'+class_name])

for child in root:
    metadata['num_docs'] += 1
    for paragraph in child:
        if paragraph.text is None:
            continue
        sentence_list = nltk.tokenize.sent_tokenize(paragraph.text,"portuguese")
        for i,s in enumerate(sentence_list):
            word_list = word_tokenize(s,"portuguese")
            metadata['num_words'] += len(word_list)
            for word in word_list:
                write_to_file(out,[word,'O'])
            if i+1 != len(sentence_list):
                out.write('\n') #sentences separated by newline
        for data in paragraph:
            skip_element = False
            empty_tag = False
            if data.tag == 'EM':
                word_list = word_tokenize(data.text,"portuguese")
                if 'CATEG' not in data.attrib.keys():
                    empty_tag = True
                else:
                    class_name = data.attrib['CATEG']
            elif data.tag == 'ALT':
                #annotate only the first alternative
                first_child = list(data)[0]
                word_list = word_tokenize(first_child.text,"portuguese")
                class_name = first_child.attrib['CATEG']
            else:
                skip_element = True
            if not skip_element:
                metadata['total_annotations'] += 1
                metadata['num_words'] += len(word_list)
                if not empty_tag:
                    process_entity(word_list,out,class_name)
                else:
                    for word in word_list:
                        write_to_file(out,[word,'O'])
            #process tail text
            if data.tail is not None:
                sentence_list = nltk.tokenize.sent_tokenize(data.tail,"portuguese")
                for i,s in enumerate(sentence_list):
                    word_list = word_tokenize(s,"portuguese")
                    metadata['num_words'] += len(word_list)
                    for word in word_list:
                        write_to_file(out,[word,'O'])
                    if i+1 != len(sentence_list):
                        out.write('\n') #sentences separated by newline

for key,val in metadata.items():
    write_to_file(metadata_file,[str(key),str(val)])

metadata_file.close()
out.close()
