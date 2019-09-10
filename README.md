## KothaDekha: Phoneme Classification from Speech Audio Spectrogram Images using CRNN
Phonemes are linguistic characteristics that are shared among languages but differ in their behavior between them. Much like speech recognition, extracting phonemes from audio is a difficult NLP challenge. Most systems use feature maps such as muse word embeddings, mel frequency cepstrum coefficients and vector tokenizations. But treating an NLP problem as a computer vision problem is an unconventional approach. Hence this paper aims to tackle the intriguing task of recognizing phonemes from visual representations of audio data. Spectrograms are created from the audio files, and a convolutional recurrent neural network is used to predict phonemes from them, using CTC for label alignment. Classification methods are used to identify the phonemes from the images.

###Instructions:
- install dependencies
  pytorch, matplotlib,
- donwload 3 npy dataset in format from:
  https://drive.google.com/drive/folders/1qXLfycK3BOShxJMH5_mFMDlS2PlA37_g
  3 files called "kothaddekha_xxxxxxArray_2k2sec.npy"
  and the saved model .pkl file
- run jupyter notebook:
  "kothadekha_network_seqCNN+BiLSTM_43.ipynb"
