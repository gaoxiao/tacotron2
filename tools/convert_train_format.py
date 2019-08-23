import os
from random import shuffle


def process_text(text):
    return text.replace('--s', '').strip()


def main():
    orig_path = '/home/gaoxiao/code/ai_utils/tts_data/downsampled_Siobhan_transcripts.txt'
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
            except Exception:
                print('Error format: {}'.format(l))

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
