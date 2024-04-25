# This file needs to be run in the main folder
# %%
import text
from utils import read_lines_from_file


def write_lines_to_file(path, lines, mode='w', encoding='utf-8'):
    with open(path, mode, encoding=encoding) as f:
        for i, line in enumerate(lines):
            if i == len(lines)-1:
                f.write(line)
                break
            f.write(line + '\n')

# %%


lines = read_lines_from_file(r'C:\Users\ni-user\Downloads\New folder\tts-arabic-pytorch\datatext.txt')
#lines = read_lines_from_file('./data/test-orthographic-transcript.txt')

new_lines_arabic = []
new_lines_phonetic = []
new_lines_buckw = []

for line in lines:
    wav_name, utterance = line.split('" "')
    wav_name, utterance = wav_name[1:], utterance[:-1]
    utterance = utterance.replace("a~", "~a") \
                         .replace("i~", "~i") \
                         .replace("u~", "~u") \
                         .replace(" - ", " ")

    
    utterance_arab = text.arabic_to_buckwalter(utterance)
    utterance_phon = text.arabic_to_phonemes(utterance)

    line_new_ara = f'"{wav_name}" "{utterance_arab}"'
    new_lines_arabic.append(line_new_ara)

    line_new_pho = f'"{wav_name}" "{utterance_phon}"'
    new_lines_phonetic.append(line_new_pho)

    line_new_buckw = f'"{wav_name}" "{utterance}"'
    new_lines_arabic.append(line_new_buckw)


# %% train

write_lines_to_file('./data/SA/train_arab.txt', new_lines_arabic)
write_lines_to_file('./data/SA/train_phon.txt', new_lines_phonetic)
write_lines_to_file('./data/SA/train_buckw.txt', new_lines_buckw)

# %% test

# write_lines_to_file('./data/test_arab.txt', new_lines_arabic)
# write_lines_to_file('./data/test_phon.txt', new_lines_phonetic)
# write_lines_to_file('./data/test_buckw.txt', new_lines_buckw)
