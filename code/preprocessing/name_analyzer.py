import re
import nltk

names = ['Marcus Gheeraerts', 'Antônio Calmon', 'Anjo Gabriel', 'Santo Amaro',
         'Ramón Villeda Morales', 'Alberto Fujimori', 'Carlo Rossi', 'Hermano Vianna',
         'Leo Robin', 'Gilberto Simoni', 'Roberto Matta', 'Isaac Albéniz',
         'Francisco Fernandes', 'José Maria Lisboa', 'Sérgio Mamberti',
         'Amanda Berry', 'Paul Quinn', 'Ignacio Warnes', 'Goran Ivanisevic',
         'Steve Johnson', 'Assis Silva', 'Eduardo Saverin', 'MC Rashid', 'Marco Popílio Lenas',
         'Artur Portela', 'Emilio Larrosa', 'Mário Andreazza', 'Cruz Ferreira',
         'Bernardo Pinto Abrunhosa', 'Maria Cláudia', 'Paulo Alexandrovich',
         'Rodrigo Oliveira', 'Tarcisio Bertone']

text = open("debug_names.txt", "r").readlines()

names_re = '('+'|'.join(names)+')'

name_ergonyms = []
total_ergonyms = []

name_prev_words = []
total_prev_words = []

for line in text:
    m = re.search(r'\[(.*)\]\s(.*)\[(.*)\]', line)
    if m is None:
        continue
    ergonyms = m.groups()[2]
    ergonyms = ergonyms.split(',')
    previous_words = m.groups()[0]
    previous_words = previous_words.split(',')
    total_ergonyms += ergonyms
    total_prev_words += previous_words
    if re.match(r''+names_re, m.groups()[1]):
        name_ergonyms += ergonyms
        name_prev_words += previous_words

names_fd = nltk.FreqDist(name_ergonyms)
total_fd = nltk.FreqDist(total_ergonyms)

name_prev_fd = nltk.FreqDist(name_prev_words)
total_prev_fd = nltk.FreqDist(total_prev_words)

print("Name ergonyms frequency")
print(names_fd.most_common()[:15])
print("Total ergonyms frequency")
print(total_fd.most_common()[:15])

print("\nName previous words frequency")
print(name_prev_fd.most_common()[:15])
print("Total previous words frequency")
print(total_prev_fd.most_common()[:15])
