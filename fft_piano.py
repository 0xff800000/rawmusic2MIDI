#!/usr/bin/env python

from midiutil import MIDIFile

T_music = 120 # Duration of the music [seconds]
T_window = 0.5 # Duration of a window [seconds]
N_windows = T_music/T_windows
Tempo = T_window * 60

## Write translation table : Note <-> frequency

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

for i, pitch in enumerate(degrees):
    MyMIDI.addNote(track, channel, pitch, time + i * T_window, T_window, volume)

with open("major-scale.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)
