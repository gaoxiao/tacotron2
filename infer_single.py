import warnings

warnings.filterwarnings("ignore")

import sys

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
# hparams.gate_threshold = 0.1

checkpoint_path = "tacotron2_statedict.pt"
# checkpoint_path = "outdir/checkpoint_12500"
# checkpoint_path = "outdir/saved_10000"
#checkpoint_path = "outdir_self_data/saved_170000"

model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

waveglow_path = 'waveglow_256channels.pt'
# waveglow_path = 'waveglow/checkpoints1/saved_356000'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()

text = 'Littlelights is awesome!'
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

# text_list = [
#     "Read loudly, and be a super hero!",
#     "Join me to learn some words.",
# ]
# sequence_list = [np.array(text_to_sequence(text, ['english_cleaners']))[None, :] for text in text_list]
# sequence_list = torch.autograd.Variable(torch.from_numpy(sequence_list)).cuda().long()

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
data = audio[0].data.cpu().numpy().astype(np.float32)
scipy.io.wavfile.write('audio_output/{}.wav'.format(text), hparams.sampling_rate, data)
