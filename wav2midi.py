import wave,math,sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import numpy as np
from midiutil import MIDIFile
from pylab import*
from scipy.io import wavfile


## Write translation frequency function : Note <-> frequency
def freq2note(f):
    if f == 0: return 0
    return int(69+np.round(12*np.log2(f/440)))

def note2freq(n):
    return 440*(2**(1/12))**(n-69)

def removeRedundant(f,sxx,notes):
    unique_notes = np.unique(np.array(notes)).tolist()
    sxx2 = np.zeros((len(unique_notes),sxx.shape[1]))
    f2 = np.array([note2freq(a) for a in unique_notes])

    for i,n in enumerate(notes):
        index = unique_notes.index(n)
        sxx2[index,:] += sxx[i,:]
    return f2,sxx2,unique_notes

## Open and show signal
path = sys.argv[1]
fs, x = wavfile.read(path)
filename = '.'.join(path.split('.')[:-1])
plt.plot(x)
plt.show()

## Fliter High pass signal
# b = signal.firwin(101, cutoff=10, fs=fs, pass_zero=False)
# x = signal.lfilter(b, [1.0], x)

## Create Spectrogram
T = 2000
T_window = 50
# f, t, Sxx = signal.spectrogram(x, fs)
# f, t, Sxx = signal.spectrogram(x, fs, nfft=88,nperseg=88)
# f, t, Sxx = signal.spectrogram(x, fs, return_onesided=True, nfft=256)
f, t, Sxx = signal.spectrogram(x, fs, return_onesided=True, nperseg=int(fs*T/1000), noverlap = int(fs*(T-T_window)/1000))
# f, t, Sxx = signal.spectrogram(x, fs, return_onesided=True, nfft=128 , nperseg=int(fs*T/1000), noverlap = int(fs*(T-T_window)/1000))

## Remove redundant notes
notes =[freq2note(a) for a in f]
print(notes)
print(np.unique(np.array(notes)))
f, Sxx, notes = removeRedundant(f,Sxx,notes)


## Rescale signal log
max_out = 255
dy = np.max(Sxx)-np.min(Sxx)
Sxx -= np.min(Sxx)
Sxx = max_out/dy * Sxx
Sxx = np.diff(Sxx)
Sxx += 1
k = max_out/np.log(max_out)
Sxx = k*np.log(Sxx)

print(np.max(Sxx))

print(f.shape)
print(Sxx.shape)
print(f)
plt.plot(f,Sxx[:,0])
plt.show()
plt.pcolormesh(t, f, Sxx)
# plt.imshow(Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

T_music = t[-1] # Duration of the music [seconds]
T_window = T_window / 1000 # Duration of a window [seconds]
N_windows = T_music/T_window
Tempo = int(1/T_window * 60)

print("Tempo {} bpm".format(Tempo))
print("T_music {} seconds".format(T_music))
print("T_window {} seconds".format(T_window))


## Decompose sigmal procedure
# 1. Decompose the signal into N windows
# 2. for each window[i]:
# 3.    compute fft window : F(f)
# 4.    compute corresponding notes and write it down
#       MIDIFile.addNote(track, channel, pitch=freq2Note(f), time=i*T_window, duration=T_window, volume=|F(f)|)

track    = 0
channel  = 0
time     = 0    # In beats
duration = 1    # In beats
tempo    = Tempo   # In BPM
volume   = 100  # 0-127, as per the MIDI standard

MyMIDI = MIDIFile(1)  # One track
MyMIDI.addTempo(track, time, tempo)

for i in range(Sxx.shape[1]):
    s = Sxx[:,i]
    notes_in_window = np.zeros(Sxx.shape[0])
    for j,volume in enumerate(s):
        if notes[j]<21 or notes[j]>108 : continue
        if volume < 0: continue
        notes_in_window[j] += volume

    for j,volume in enumerate(notes_in_window):
        if math.isnan(volume): continue
        volume = int(volume)
        if volume == 0: continue
        # print(notes[j]," ",volume)
        MyMIDI.addNote(track, channel, notes[j], time + i  , 1, volume)

with open(filename+".mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)
