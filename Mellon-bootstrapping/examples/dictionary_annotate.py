import nltk
import re

"""
    Annotate dataset with dictionary approach
    dictionary: list of names
    words_to_add: new names identified
    text: text to annotate
"""


def is_valid_match(indexes, start, end):
    x = [(s, e) for (s, e) in indexes if (start >= s and start <= e) or (end <= e and end >= s)]
    return len(x) == 0


def dictionary_annotate(re_list, conll_text):
    match_indexes = []
    for regex in re_list:
        matches = regex.finditer(conll_text)
        for m in matches:
            if is_valid_match(match_indexes, m.start(), m.end()):
                match_indexes.append((m.start(), m.end()))
    return match_indexes


"""
    Annotate using IOBES scheme
"""


def annotate_conll(conll_text, indexes):
    result = ''

    indexes = sorted(indexes, key=lambda x: x[0])
    last_end = 0
    for (s, e) in indexes:
        result += process_sub_text(conll_text[last_end:s], 'O')
        result += process_tagged_sub_text(conll_text[s:e])
        #special case when ... after a name
        if e < len(conll_text):
            if conll_text[e] == '\n':
                last_end = e+1
            else:
                last_end = e
    result += process_sub_text(conll_text[last_end:], 'O')
    return result


def process_sub_text(text, tag):
    if len(text) == 0:
        return ''
    if text[-1] == '\n':
        text = text[:-1]
    lines = text.split('\n')
    result = ''
    for line in lines:
        if line == '':
            result += '\n'
        else:
            result += line+'\t'+tag+'\n'
    return result

# Follow IOB2
def process_tagged_sub_text(text):
    lines = text.split('\n')
    if len(lines) == 1:
        return lines[0]+'\tB-PER\n'
    result = lines[0]+'\tB-PER\n'
    for line in lines[1:]:
        result += line+'\tI-PER\n'
    return result


"""
    Inputs:
    raw_text_filename - filename of raw text to Annotate
    initial_name_list
    additional_name_list
    output_filename - filename of output file

    Process one line of input at a time
"""


def annotate(raw_text_filename, name_list, output_filename):
    print("Annotating raw text with names.")
    f_in = open(raw_text_filename, 'r')
    f_out = open(output_filename+'.tmp', 'w')
    name_list = sorted(name_list, key=len, reverse=True)
    regex_list = [re.compile(r''+'\n'.join(name.split(' '))+'') for name in name_list]

    counter = 0
    buffer_string = ''

    for line in f_in:
        if line == '\n':
            counter += 1
            ids = dictionary_annotate(regex_list, buffer_string)
            f_out.write(annotate_conll(buffer_string, ids)+'\n')
            if counter % 10000 == 0:
                print("Annotating... {:16} sentences processed".format(counter))
            buffer_string = ''
        else:
            buffer_string += line
    # last sentence
    ids = dictionary_annotate(regex_list, buffer_string)
    f_out.write(annotate_conll(buffer_string, ids)+'\n')

    f_in.close()
    f_out.close()
    f_out_t = open(output_filename+'.tmp', 'r')
    f_out = open(output_filename, 'w')
    counter = 1
    for line in f_out_t:
        line_s = line.split('\t')
        if line == '\n':
            counter = 1
        else:
            line = "{} {} {} {} {}".format(counter, line_s[0], 'O', 'O', line_s[1])
            counter += 1
        f_out.write(line)

    f_out_t.close()
    f_out.close()

    print("\nAnnotation complete")
    return output_filename
