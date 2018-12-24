import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import numpy as np
from midiutil import MIDIFile

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

f, t, Sxx = signal.spectrogram(x, fs, return_onesided=True)
print(f.shape)
print(Sxx.shape)
print(f)
plt.plot(f,Sxx[:,0])
plt.show()
plt.pcolormesh(t, f, Sxx)
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
T_window = t[1]-t[0] # Duration of a window [seconds]
N_windows = T_music/T_window
Tempo = int(N_windows * 60 / 10)

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

max_Sxx = np.max(Sxx)

for i in range(Sxx.shape[1]):
    s = Sxx[:,i]
    notes_in_window = []
    for j,volume in enumerate(s):
        if notes[j]<21 or notes[j]>108 or notes[j] in notes_in_window: continue
        vol = int(127*(volume/max_Sxx))
        if vol == 0 or vol < 50: continue
        # print(notes[j]," ",vol)
        notes_in_window.append(notes[j])
        MyMIDI.addNote(track, channel, notes[j], time + i , T_window, vol)

# print("end")

# for i, pitch in enumerate(degrees):
#     MyMIDI.addNote(track, channel, pitch, time + i * T_window, T_window, volume)

with open("major-scale.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)
