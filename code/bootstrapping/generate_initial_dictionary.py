import argparse
import re


def increment_dict_count(d, key, inc):
    if key in d:
        d[key] += inc
    else:
        d[key] = inc


"""
    Extract capitalized sequences that follow the format:
        [Capitalization Sequence],[ergonym]
        ex: [José Socrates],[presidente]
"""

parser = argparse.ArgumentParser(description='Script obtains list of names from raw data.')
parser.add_argument("input_filename", help="relative path to trained model")
parser.add_argument("output_filename", help="relative path to output file")
parser.add_argument("text_type", help="Textual type to process", choices=['wikipedia', 'news'])
parser.add_argument("-t", "--threshold", help="Minimum number of occurrences", type=int, default=3)
parser.add_argument("-d", "--debug", help="Debug flag", action="store_true")
parser.add_argument("-i", "--inspect", help="Match all ergonyms without restriction",
                    action="store_true")
args = parser.parse_args()

CAP_SEQUENCE_MIN_LENGTH = 2
input_file = open(args.input_filename, 'r')

data = input_file.read()

occurrences = dict()
detected_ergonyms = dict()
previous_words = dict()

ergonym_raw_list = ['presidente', 'jogador', 'treinador', 'deputado', 'arcebispo', 'pastor',
                    'fundador', 'companheiro', 'licenciado', 'residente', 'dinamizador',
                    'especialista', 'candidato', 'jornalista', 'comerciante', 'autor',
                    'conselheiro', 'professor', 'ministro', 'comandante', 'chefe',
                    'guitarrista', 'atriz', 'actor', 'vocalista', 'cantor', 'cantora',
                    'produtor', 'físico', 'descobridor', 'piloto',
                    'almirante', 'encenador', 'arquiteto', 'governador', 'cardeal', 'bispo',
                    'músico', 'irmão', 'irmã', 'filho', 'filha', 'pai', 'mãe', 'avô', 'avó',
                    'tio', 'tia', 'primo', 'prima', 'marido', 'mulher'
                    ]

possible_previous_words = ['por', 'como', 'segundo', 'iorquino', 'canadense',
                           'alemão', 'brasileiro', 'brasileira', 'português', 'portuguesa',
                           'estadunidense', 'americana', 'americano', 'coreano', 'francês',
                           'espanhol', 'espanhola', 'fracesa', 'alemão', 'alemã', 'inglês',
                           'inglesa', 'britânico', 'britânica', 'chileno', 'chilena', 'argentino',
                           'peruano', 'peruana', 'ucraniano', 'ucraniana', 'islandesa', 'islandês'
                           ]
# consider 1st capital letter
ergonym_list = []
for ergonym in ergonym_raw_list:
    ergonym_list += [ergonym, ergonym.capitalize()]
possible_previous_words_list = []

for w in possible_previous_words:
    possible_previous_words_list += [w, w.capitalize()]

# include ergonyms as possible previous words
possible_previous_words_list += ergonym_list

if args.text_type == 'wikipedia':
    if args.inspect:
        ergonym_re = '([^\s]+)'
        previous_word_re = '(^|\w+)'
    else:
        ergonym_re = '('+'|'.join(ergonym_list)+')'
        previous_word_re = '('+'|'.join(possible_previous_words_list)+')'

    matches = re.finditer(r''+previous_word_re+'\s([A-Z]\w+(?:\s[A-Z]\w+)+),\s'+ergonym_re, data,
                          flags=re.MULTILINE)

    for match in matches:

        name = match.groups()[1]
        if args.debug:
            previous_word = match.groups()[0]
            ergonym = match.groups()[2]

        if len(name.split(' ')) < CAP_SEQUENCE_MIN_LENGTH:
            continue

        increment_dict_count(occurrences, name, 1)

        if args.debug:
            increment_dict_count(detected_ergonyms, name, [ergonym])
            increment_dict_count(previous_words, name, [previous_word])

    output_file = open(args.output_filename, "w")

    for name, num_occ in occurrences.items():
        if num_occ < args.threshold:
            continue
        else:
            if args.debug:
                output_file.write("[{}] {}, [{}]\n".format(', '.join(
                    previous_words[name]), name, ', '.join(detected_ergonyms[name])))
            else:
                output_file.write(name+'\n')

elif args.text_type == 'news':
    specific_news_ergonyms = ['ex-líder', 'especializado', 'especializada', 'apelou', 'desafiou',
                              'militante', 'administrador', 'vice-ministro', 'presidiu', 'homenageou',
                              'criou', 'alertou', 'mostrou']

    if args.inspect:
        ergonym_re = '([^\s]+)'
    else:
        ergonym_re = '('+'|'.join(ergonym_list+specific_news_ergonyms)+')'

    matches = re.finditer(r'([A-Z]\w+(?:\s[A-Z]\w+)+),\s'+ergonym_re, data,
                          flags=re.MULTILINE)

    for match in matches:

        name = match.groups()[0]
        if args.debug:
            ergonym = match.groups()[1]

        if len(name.split(' ')) < CAP_SEQUENCE_MIN_LENGTH:
            continue

        increment_dict_count(occurrences, name, 1)

        if args.debug:
            increment_dict_count(detected_ergonyms, name, [ergonym])

    output_file = open(args.output_filename, "w")

    for name, num_occ in occurrences.items():
        if num_occ < args.threshold:
            continue
        else:
            if args.debug:
                output_file.write("{}, [{}]\n".format(
                    name, ', '.join(detected_ergonyms[name])))
            else:
                output_file.write(name+'\n')
