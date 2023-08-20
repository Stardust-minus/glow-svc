# Copilot inside warning
import argparse
import librosa
import numpy as np
import soundfile as sf
import torch
from model import Generator
from vocoder import Vocoder


def load_models(model_path, vocoder_path):
    generator = Generator()
    generator.load_state_dict(torch.load(model_path))
    generator.eval()
    generator.cuda()

    vocoder = Vocoder()
    vocoder.load_state_dict(torch.load(vocoder_path))
    vocoder.eval()
    vocoder.cuda()

    return generator, vocoder


def generate_audio(generator, vocoder, x, f0, speaker):
    with torch.no_grad():
        x = x.cuda()
        f0 = f0.cuda()
        speaker = speaker.cuda()

        mel_flow, pred_f0 = generator.module(x, f0=f0, g=speaker, gen=True, glow=True)
        y_flow = vocoder.spec2wav(mel_flow.squeeze(0).transpose(0, 1).cpu().numpy(),
                                  f0=pred_f0[0, 0, :].cpu().numpy())

        return y_flow


def main(args):
    generator, vocoder = load_models(args.model_path, args.vocoder_path)

    x, sr = librosa.load(args.input_path, sr=16000)
    x = librosa.util.fix_length(x, 16000 * 5)
    x = np.expand_dims(x, axis=0)
    f0 = np.zeros((1, 1, 256))
    speaker = np.zeros((1, 1))

    y = generate_audio(generator, vocoder, x, f0, speaker)

    sf.write(args.output_path, y, sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate audio using a pre-trained model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model.')
    parser.add_argument('--vocoder_path', type=str, required=True, help='Path to the pre-trained vocoder.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input audio file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output audio file.')
    args = parser.parse_args()

    main(args)
