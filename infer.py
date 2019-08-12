import warnings
warnings.filterwarnings("ignore")

import sys
import time

import matplotlib.pylab as plt
import scipy

sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from train import load_model
from text import text_to_sequence


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')


hparams = create_hparams()
hparams.sampling_rate = 22050

checkpoint_path = "tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

waveglow_path = 'waveglow_256channels.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()

# text = "Read loudly, and be a super hero!"
# sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
# sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

text_list = [
    "Join me to learn some words.",
    "Tom likes drums",
    "Tom does not like kites",
    "Now you try!",
    "Wow, you matched all of the words!",
    "Great work reading those words!",
    "It's time for some vocabulary!",
    "Hi, Andrew here!",
    "Let's review what you said",
    "baseball",
    "Sorry. I did not hear you. Could you say it louder?",
    "Hello, Red Beetle",
    "HIllary and Henry are hiding in their school. Can you find them?",
]
for text in text_list:
    start_time = time.time()
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    print("--- text to seq: %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    print("--- tacotron2 %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    print("--- waveglow %s seconds ---" % (time.time() - start_time))

    data = audio[0].data.cpu().numpy().astype(np.float32)
    scipy.io.wavfile.write('output/{}.wav'.format(text), hparams.sampling_rate, data)
