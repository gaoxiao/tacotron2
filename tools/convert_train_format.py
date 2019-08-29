import os
import string
from random import shuffle


def process_text(text):
    text = text.replace('--s', '')
    text = text.replace('-', ',')
    while True:
        start = text.find('(')
        end = text.find(')')
        if start != -1 and end != -1:
            text = text[:start] + text[end + 1:]
        else:
            break
    text = text.strip()
    if text[-1] not in string.punctuation:
        text = text + '.'
    return text


def main():
    orig_path = '/home/xiao/code/ai_utils/tts_data/downsampled_Sam_transcripts.txt'
    if not os.path.isfile(orig_path):
        print('File not found: {}, skipping'.format(orig_path))
        return

    to_calculate = []
    with open(orig_path) as f:
        for l in f:
            try:
                text, path = l.split('|')
                text = process_text(text)
                path = '/home/gaoxiao/code/ai_utils/tts_data/{}'.format(path.strip())
                if len(text.split()) > 20:
                    continue
                to_calculate.append('|'.join((path, text)))
            except Exception as e:
                print('Error format: {}'.format(l))
                print(e)

    size = len(to_calculate)
    shuffle(to_calculate)
    train_size = int(size * 0.9)
    train_set = to_calculate[:train_size]
    test_set = to_calculate[train_size:]

    def write_file(line_set, file_path):
        with open(file_path, 'w') as f:
            for l in line_set:
                f.write('{}\n'.format(l))

    write_file(train_set, 'train.txt')
    write_file(test_set, 'test.txt')


if __name__ == '__main__':
    main()
