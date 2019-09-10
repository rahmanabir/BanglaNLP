import os
import glob
import wave
import pylab
import csv
import matplotlib
import librosa
import librosa.display
import numpy as np

savePath = 'data/spectographs'

# creating spectrograms from audio info
def graph_spectrogram(wav_file):
    wav_file = 'data/crblp/wav/'+wav_file
    sound_info, frame_rate = get_wav_info(wav_file)
    filename = wav_file.replace("data/crblp/wav/", "")
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.axis('off')
    pylab.savefig(savePath+'/'+filename+'.jpg', bbox_inches="tight", frameon='false')
    pylab.close()

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
    wav_file = 'data/crblp/wav/'+wav_file
    filename = wav_file.replace("data/crblp/wav/", "")
    sig, fs = librosa.load(wav_file)
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    pylab.savefig(savePath+'_Mel/'+filename+'.jpg', bbox_inches=None, pad_inches=0)
    pylab.close()

#white
#black
#yellow
#orange
#red
#green
#purple
#blue
#pink

# main function
if __name__ == '__main__':

    wav_range = []
    with open('dekhabet_dataLabelsRanged.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            text = row[0]
            wav_range.append(text)
    csvFile.close()
    wav_range.pop(0)
    # rangeFile = open("RangedAudiofileList.txt", "r")
    # wav_range = rangeFile.readlines()
    c = len(wav_range)
    # print(c)
    # print(wav_range)
    i = 0
    while i < c:
        wav_file = wav_range[i].rstrip('\n')
        graph_spectrogram(wav_file+'.wav')
        graph_melcepstrum(wav_file+'.wav')
        i += 1
        if i % 50 == 0:
            print(i,'graphs drawn in data/ dir')

    # rangeFile.close()
