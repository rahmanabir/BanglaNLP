
import os
import glob
import wave
import pylab

savePath = '../data/spectographs/'
# creating spectrograms from audio info


def graph_spectrogram(wav_file):

    filename = wav_file.replace("../data/crblp/wav/", "")
    print(filename)
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.specgram(sound_info, Fs=frame_rate)
    # spec, freq, t, im = pylab.specgram(sound_info, Fs=frame_rate)
    pylab.axis('off')
    pylab.savefig(os.path.splitext(savePath+filename)[
                  0], bbox_inches="tight", frameon='false')

# extracts info from wav file


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(44100*10)  # Limit set to 10 seconds
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


# main function
if __name__ == '__main__':

    rangeFile = open("RangedAudiofileList.txt", "r")
    wav_range = rangeFile.readlines()

    i = 0
    while i <= len(wav_range):
        wav_file = '../' + wav_range[i].rstrip('\n')
        graph_spectrogram(wav_file)
        i += 1

    rangeFile.close()
