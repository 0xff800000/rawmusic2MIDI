ffmpeg -i $1 -acodec pcm_s16le -ar 16000 -ac 1 $1.wav

python3.6 wav2midi.py $1.wav
rm $1.wav

