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
        last_end = e+1
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


def process_tagged_sub_text(text):
    lines = text.split('\n')
    if len(lines) == 1:
        return lines[0]+'\tS-PER\n'
    result = lines[0]+'\tB-PER\n'
    for line in lines[1:-1]:
        result += line+'\tI-PER\n'
    result += lines[-1]+'\tE-PER\n'
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
    f_out = open(output_filename, 'w')
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
                print("Annotating... {:16} sentences processed".format(counter), end='\r')
            buffer_string = ''
        else:
            buffer_string += line
    # last sentence
    ids = dictionary_annotate(regex_list, buffer_string)
    f_out.write(annotate_conll(buffer_string, ids)+'\n')

    f_in.close()
    f_out.close()
    print("\nAnnotation complete")
    return output_filename


if __name__ == "__main__":
    d = ['Manuel Silva', 'José Costa Barbosa']
    w = ['Carlos Monteiro', 'Carlos']
    text = """Carlos Monteiro do Togo em estado grave, mas estável O José Costa Barbosa foi transferido de urgência sábado de Angola para a unidade hospitalar sul-africana para ser sujeito a uma delicada intervenção cirúrgica. "Estamos satisfeitos com os progressos do Sr. Olilale. Actualmente está na Unidade de Cuidados Intensivos de Traumatologia, onde vai permanecer até que estejamos mais optimistas com a sua condição", explicou o doutor Elias Degiannis, precisando que jogador encontra-se sedado e com respiração assistida. "Ontem à noite [sábado] foram momentos difíceis para todos, particularmente para a família, antes da operação. Estamos em contacto constante com a família em França e no Togo e continuaremos a proporcionar informações sobre a sua evolução", acrescentou. O médico disse ainda que dentro de dois ou três dias será possível prestar informação mais detalhada quanto ao estado de saúde do futebolista, quando este puder respirar pelos seus próprios meios, e referiu que este conseguiu mover as pernas quando deu entrada no hospital. Manuel Silva foi atingido por uma bala na parte inferior das costas durante o ataque a que foi sujeita a comitiva da selecção do Togo, na fronteira entre o Congo e Angola, já no interior do enclave de Cabinda. O ataque, reivindicado pela guerrilha separatista da Frente para a Libertação do Enclave de Cabinda (FLEC), causou a morte ao condutor do autocarro, de origem angolana, e a dois membros da delegação togolesa, tendo outros seis ficado feridos. O Carlos viajava para Cabinda, território onde tem previsto para segunda-feira um encontro com a selecção do Gana, referente ao Grupo B da Taça das Nações Africanas de futebol de Angola (CAN2010)."""
    # text = """Carlos Monteiro do Togo em estado grave, mas estável O José Costa Barbosa"""
    p = text_to_conll(text)
    indexes = dictionary_annotate(d+w, p)
    annotated = annotate_conll(p, indexes)
    print(annotated, end='')
