import os
import subprocess

from tqdm import tqdm


def downsample_file(in_path, out_path):
    try:
        sr = 8000
        if not os.path.isfile(out_path):
            FNULL = open(os.devnull, 'w')
            completed = subprocess.run(['sox', in_path, '-r', '22050', out_path], stdout=FNULL,
                                       stderr=subprocess.STDOUT)
            # if err:
            #     print(output, err)
            # p = subprocess.Popen(['sox', in_path, out_path, 'remix', '1-2'], stdout=FNULL,
            #                      stderr=subprocess.STDOUT)
            # completed = subprocess.run(['sox', in_path, out_path, 'remix', '1-2'], stdout=FNULL,
            #                            stderr=subprocess.STDOUT)
            # print('returncode:', completed.returncode)
            # output, err = p.communicate()
            # if err:
            #     print(output, err)
    except Exception as e:
        print(e)
        print('wrong format: {}'.format(in_path))


def gene(data_path, out_file):
    data = []
    paths = os.listdir(data_path)
    for f in tqdm(paths):
        if not f.endswith('wav'):
            continue
        if f.startswith('downsampled_'):
            continue
        audio_path = f
        downsampled = 'downsampled_{}'.format(audio_path)
        audio_path = os.path.join(data_path, audio_path)
        downsampled = os.path.join(data_path, downsampled)
        downsample_file(audio_path, downsampled)
        text_path = '{}.trn'.format(audio_path)
        with open(text_path, 'r') as text_file:
            text_path = text_file.read()
            text_path = text_path.strip()
        text_path = os.path.join(data_path, text_path)
        with open(text_path, 'r') as text_file:
            text = text_file.readlines()
            text = text[1].strip()
            data.append('{}|{}.\n'.format(downsampled, text))
    print('len: {}'.format(len(data)))
    with open(out_file, 'w') as out:
        out.writelines(data)


def main():
    gene('/home/gaoxiao/code/tacotron2/cn/data_thchs30/train', 'train.txt')
    gene('/home/gaoxiao/code/tacotron2/cn/data_thchs30/test', 'test.txt')


if __name__ == '__main__':
    main()
