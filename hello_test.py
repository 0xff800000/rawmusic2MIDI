import matplotlib.pyplot as plt
import numpy as np
import wave
import math
from scipy import signal
import numpy as np
from midiutil import MIDIFile

from pylab import*
from scipy.io import wavfile

fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = np.sqrt(2)*4e3*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*2e3*time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier


# spf = wave.open('peer.wav','r')
#
# #Extract Raw Audio from Wav File
# x = spf.readframes(-1)
# x = np.fromstring(x, 'Int16')
# fs = spf.getframerate()

# fs, x = wavfile.read('peer.wav')
# fs, x = wavfile.read('notorious.wav')
# fs, x = wavfile.read('ave.wav')
# fs, x = wavfile.read('gc.wav')
fs, x = wavfile.read('hello.wav')
x = x / np.max(np.abs(x))
# x = np.mean(x, axis=1)
plt.plot(x)
plt.show()
# fs = 8000

T = 512
T_window = 50
# f, t, Sxx = signal.spectrogram(x, fs)
# f, t, Sxx = signal.spectrogram(x, fs, nfft=88,nperseg=88)
# f, t, Sxx = signal.spectrogram(x, fs, return_onesided=True, nfft=256)
f, t, Sxx = signal.spectrogram(x, fs, return_onesided=True, nperseg=int(fs*T/1000), noverlap = int(fs*(T-T_window)/1000))
# f, t, Sxx = signal.spectrogram(x, fs, return_onesided=True, nfft=128 , nperseg=int(fs*T/1000), noverlap = int(fs*(T-T_window)/1000))

# Sxx += 1


dy = np.max(Sxx)-np.min(Sxx)
Sxx -= np.min(Sxx)
Sxx = 255/dy * Sxx
Sxx += 1

k = 255/np.log(255)
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

## Write translation frequency function : Note <-> frequency
def freq2note(f):
    if f == 0: return 0
    return int(69+np.round(12*np.log2(f/440)))

notes =[freq2note(a) for a in f]
print(notes)

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

#           C , D , E , F , G,  A , B , C
#           C , D , E , F , G, 440 , B , C , D , E , F , G , 880
degrees  = [60, 62, 64, 65, 67, 69, 71, 72]  # MIDI note number
track    = 0
channel  = 0
time     = 0    # In beats
duration = 1    # In beats
tempo    = Tempo   # In BPM
volume   = 100  # 0-127, as per the MIDI standard

MyMIDI = MIDIFile(1)  # One track
MyMIDI.addTempo(track, time, tempo)

# max_Sxx = np.max(Sxx)

for i in range(Sxx.shape[1]):
    s = Sxx[:,i]
    # notes_in_window = np.zeros(Sxx.shape[0])
    notes_in_window = {}
    for j,volume in enumerate(s):
        if notes[j]<21 or notes[j]>108 : continue
        if volume < 0: continue
        # vol = 127*(volume/np.max(s))
        # vol = 127*(volume/max_Sxx)
        # notes_in_window[j] += vol
        if notes[j] in notes_in_window:
            if notes_in_window[notes[j]] + volume <= 255:
                notes_in_window[notes[j]] += volume
            else:
                notes_in_window[notes[j]] = 255
        else:
            notes_in_window[notes[j]] = volume
        # notes_in_window[j] += k*np.log(volume)

    # note_played = []
    # for j,volume in enumerate(notes_in_window):
    for j,volume in notes_in_window.items():
        if math.isnan(volume): continue
        volume = int(volume)
        if volume < 10: continue
        print(notes[j]," ",volume)
        # if notes[j] in note_played: continue
        # note_played.append(notes[j])
        # MyMIDI.addNote(track, channel, notes[j], time + i * T_window , T_window, volume)
        # MyMIDI.addNote(track, channel, notes[j], time + i  , 1, volume)
        MyMIDI.addNote(track, channel, j, time + i  , 1, volume)

# print("end")

# for i, pitch in enumerate(degrees):
#     MyMIDI.addNote(track, channel, pitch, time + i * T_window, T_window, volume)

with open("major-scale.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)
