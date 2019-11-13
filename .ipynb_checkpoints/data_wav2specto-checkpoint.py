# import os
# import glob
import wave
import pylab
# import csv
# import matplotlib
import librosa
import librosa.display
import numpy as np

savePath = 'data/spectrograms'

# creating spectrograms from audio info
def graph_spectrogram(wav_file):
#     wav_file = 'data/crblp/wav/'+wav_file
#     sound_info, frame_rate = get_wav_info(wav_file)
#     filename = wav_file.split('/')[-1]
#     pylab.figure(num=None, figsize=(19, 12))
#     pylab.subplot(111)
#     pylab.specgram(sound_info, Fs=frame_rate)
#     pylab.axis('off')
    filename = wav_file.split('/')[-1]
    sig, fs = librosa.load(wav_file)
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    S = librosa.feature.chroma_stft(y=sig, sr=fs)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    pylab.savefig(savePath+'/'+filename+'.jpg', bbox_inches=None, pad_inches=0)
    pylab.close('all')

# extracts info from wav file
def get_wav_info(wav_file):
    # wav_file = 'data/crblp/wav/'+wav_file
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(44100*5)  # Limit set to 5 seconds
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

def graph_melcepstrum(wav_file):
#     wav_file = 'data/crblp/wav/'+wav_file
    fig = pylab.figure(figsize=(19, 12))
    filename = wav_file.split('/')[-1]
    sig, fs = librosa.load(wav_file)
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    fig.savefig(savePath+'_mel/'+filename+'.png', bbox_inches=None, pad_inches=0)
    pylab.close(fig)
    # pylab.close('all')

#white
#black
#yellow
#orange
#red
#green
#purple
#blue
#pink

import progressbar as pgrs


#############################

def openslr_convert_w2s():

    import pandas as pd
    import time

    df = pd.read_csv('data/asr_bengali/openslr_transcript.csv')
    durationrange = [2.5, 3.5]
    data_dir = 'data/asr_bengali/data/'
    count = 0
    l = len(df)
    start_time = time.time()
    print('Conversion Started at', time.ctime())
    pgrs.printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 40)
    for i, row in df.iterrows():
        pgrs.printProgressBar(i+1, l, prefix = 'Converting:', suffix = '[{}/{}]'.format(i,l), length = 50)
        wavfile = data_dir + '{}/{}.flac'.format(row['Filename'][0:2], row['Filename'])
        if row['DurationSec']>=durationrange[0] and row['DurationSec']<=durationrange[1]:
    #         w2s.graph_spectrogram(wavfile)
            graph_melcepstrum(wavfile)
            count += 1
    print('Conversion Complete at', time.ctime())
    print('Time Elapsed: ', time.strftime("%H hrs %M mins %S secs", time.gmtime(time.time() - start_time)))
    print(count, 'mel spectrograms calculated')


# openslr_convert_w2s()