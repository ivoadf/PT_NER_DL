import argparse
import re

"""
    Parse wikipedia WikiMedia format to raw text:
"""

parser = argparse.ArgumentParser(description='Script to parse WikiMedia format to raw text')
parser.add_argument("input_filename", help="relative path to WikiMedia file")
parser.add_argument("output_filename", help="relative path to output file")
args = parser.parse_args()

input_file = open(args.input_filename, 'r')
output_file = open(args.output_filename, 'w')

text = False
insideParas = False

buffer = ''

for line in input_file:
    if re.search(r'<text ', line):
        text = True
    if text:
        if re.search(r'<\/text>', line):
            text = False
            continue
        if re.match(r'{', line):
            insideParas = True
        if re.match(r'.*}\n', line):
            insideParas = False
        if line == '\n' and not insideParas:
            buffer += line
            # url encoded
            buffer = re.sub(r'&amp;', '&', buffer)
            buffer = re.sub(r'&lt;', '<', buffer)
            buffer = re.sub(r'&gt;', '>', buffer)
            # sub &quot; with "
            buffer = re.sub(r'&quot;', '"', buffer)
            # br tags
            buffer = re.sub(r'<br\s?\/>', '', buffer)
            # refs
            buffer = re.sub(r'<ref[^<]*<\/ref>', '', buffer)
            buffer = re.sub(r'<ref\s?>([^<>]|<[^r]*>)*<\/ref>', '', buffer)
            # xml tags
            buffer = re.sub(r'<[^>]*>', '', buffer)
            # images, files, categories
            buffer = re.sub(r'\[\[(Imagem|Image|imagem|image):.*\]\](?=\n)', '', buffer, flags=re.DOTALL)
            buffer = re.sub(r'{{(Imagem|Image|imagem|image).*}}', '', buffer, flags=re.DOTALL)
            buffer = re.sub(r'\[\[(Ficheiro|File):.*\]\]', '', buffer)
            buffer = re.sub(r'Ficheiro:.*', '', buffer)
            buffer = re.sub(r'\[\[(Categoria|Category):.*\]\]', '', buffer)
            # triple apostrophes and parentheses
            buffer = re.sub(r"\[\[([^\]\n]+)\]\]", r'\1', buffer)
            buffer = re.sub(r"'''", '', buffer)
            # paratheses separated by |
            buffer = re.sub(r"\[\[[^\]]*\|([\w\s\-\/\.',\(\)]+)\]\]", r'\1', buffer)
            # subheadings
            buffer = re.sub(r"={2,}(.+)", '', buffer)
            # citations and links to other languages
            buffer = re.sub(r'{{([^{}]|{{.*}})*}}', '', buffer, flags=re.DOTALL)
            buffer = re.sub(r'\([\s;]+\)', '', buffer)
            # Tables
            buffer = re.sub(r'{[^}]*}', '', buffer)
            # lists
            buffer = re.sub(r'^(\*|:)(.*)', '', buffer, flags=re.MULTILINE)
            # links left between []
            buffer = re.sub(r'\[[^\]]*\]', '', buffer)
            # html urls
            buffer = re.sub(r'https?[^\s]*', '', buffer)
            # lines starting with ;
            buffer = re.sub(r'^;.*', '', buffer, flags=re.MULTILINE)
            # sub '' with "
            buffer = re.sub(r"''", '"', buffer)
            # # at the start of sentences
            buffer = re.sub(r"^#\s", '', buffer, flags=re.MULTILINE)
            # clear extra newlines
            buffer = re.sub(r'^\n+', '\n', buffer, flags=re.MULTILINE)
            output_file.write(buffer)
            buffer = ''
        else:
            buffer += line
input_file.close()
output_file.close()
