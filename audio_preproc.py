import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os

#-------------CHANGE FOLDER NAME----------------------
input_folder = '/content/drive/MyDrive/FYP/First Review/sample audio data/'
output_folder = '/content/drive/MyDrive/FYP/First Review/sample audio mels/'


# iterate over files in
# that folder
for filename in os.listdir(input_folder):
    print("file " + filename)
    audio_file = os.path.join(input_folder, filename)
    # checking if it is a file
    if os.path.isfile(audio_file):
        print("audio file is " + audio_file)
        output_file = "out_" + filename[:len(filename) - 4]
        print(output_file)
        y, sr = librosa.load(audio_file)
        S = librosa.feature.melspectrogram(y=y, sr=16000,hop_length=200, win_length=800,  fmax=8000)
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        plt.savefig(output_folder + output_file + ".png")

        plt.show()