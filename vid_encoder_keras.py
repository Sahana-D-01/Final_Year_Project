import keras
from keras.models import Sequential
from keras.layers import Flatten, Conv3D, BatchNormalization, Bidirectional, LSTM, Reshape
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import TimeseriesGenerator

import numpy as np
# import matplotlib.pyplot as plt
import h5py
import os

from argparse import ArgumentParser
from glob import glob
from os import path

import tensorflow as tf

parser = ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
# parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=16, type=int)
parser.add_argument("--speaker_root", help="Root folder of Speaker", required=True)
# parser.add_argument("--resize_factor", help="Resize the frames before face detection", default=1, type=int)
parser.add_argument("--speaker", help="Helps in preprocessing", required=False, choices=["chem", "chess", "hs", "dl", "eh","s1"])

args = parser.parse_args()

def createModel():
      model = Sequential()
      model.add(Conv3D(3, kernel_size=(5,5,5), strides=(1,2,2), activation='relu', padding='same', input_shape=(40,96,96,3)))
      model.add(BatchNormalization())
      model.add(Conv3D(3, kernel_size=(3,3,3), strides=(1,1,1), activation='relu', padding='same'))
      model.add(BatchNormalization())
      model.add(Conv3D(32, kernel_size=(3,3,3), strides=(1,1,1), activation='relu', padding='same'))
      model.add(BatchNormalization())
      model.add(Conv3D(32, kernel_size=(3,3,3), strides=(1,2,2), activation='relu', padding='same'))
      model.add(BatchNormalization())
      model.add(Conv3D(32, kernel_size=(3,3,3), strides=(1,1,1), activation='relu', padding='same'))
      model.add(BatchNormalization())
      model.add(Conv3D(64, kernel_size=(3,3,3), strides=(1,1,1), activation='relu', padding='same'))
      model.add(BatchNormalization())
      model.add(Conv3D(64, kernel_size=(3,3,3), strides=(1,2,2), activation='relu', padding='same'))
      model.add(BatchNormalization())
      model.add(Conv3D(64, kernel_size=(3,3,3), strides=(1,1,1), activation='relu', padding='same'))
      model.add(BatchNormalization())
      model.add(Conv3D(128, kernel_size=(3,3,3), strides=(1,1,1), activation='relu', padding='same'))
      model.add(BatchNormalization())
      model.add(Conv3D(128, kernel_size=(3,3,3), strides=(1,2,2), activation='relu', padding='same'))
      model.add(BatchNormalization())
      model.add(Conv3D(128, kernel_size=(3,3,3), strides=(1,1,1), activation='relu', padding='same'))
      model.add(BatchNormalization())
      model.add(Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1), activation='relu', padding='same'))
      model.add(BatchNormalization())
      model.add(Conv3D(256, kernel_size=(3,3,3), strides=(1,2,2), activation='relu', padding='same'))
      model.add(BatchNormalization())
      model.add(Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1), activation='relu', padding='same'))
      model.add(BatchNormalization())
      model.add(Conv3D(512, kernel_size=(3,3,3), strides=(1,1,1), activation='relu', padding='same'))
      model.add(BatchNormalization())
      model.add(Conv3D(512, kernel_size=(3,3,3), strides=(1,4,4), activation='relu', padding='same'))
      model.add(BatchNormalization())
      model.add(Flatten())
      model.add(Reshape(target_shape=(40,512)))
      model.add(Bidirectional(LSTM(256, return_sequences=True)))
      model.compile(loss='mean_absolute_error', optimizer='adam')
      return model

def frame_sequence_generator(generator):
    while True:
        frame_batch = generator.next()
        frame_batch = frame_batch.reshape((-1, 40) + frame_batch.shape[1:])
        target_batch = np.zeros((64,1))
        yield(frame_batch, target_batch)

def load_dataset(filepath):
      frameSequenceLength = 40
      batchSize = 64
      height = 96
      width = 96

      datagen = ImageDataGenerator()
      return datagen.flow_from_directory(
            filepath,
            target_size=(height,width),
            color_mode='rgb',
            batch_size=batchSize*frameSequenceLength,
            class_mode=None,
            classes=None,
            shuffle=False
      )

    #   print(dataset.class_indices)
    #   print(dataset.class_mode)

    #   return TimeseriesGenerator(
    #         filepath,
    #         targets=None,
    #         length=40,
    #         batch_size=64
    #   )

def main(args):
    # print('Started processing for {} with {} GPUs'.format(args.speaker_root, args.ngpu))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    #create file path to all the images
    file_path = glob(path.join(args.speaker_root, 'preprocessed'))

    #load all the frames into dataset
    dataset = load_dataset(file_path[0])
    print(type(dataset))
    print(len(dataset))
    print(dataset.image_shape)
 
    frame_seq_dataset = frame_sequence_generator(dataset)

    print(type(frame_seq_dataset))
    model = createModel()
    # # model.summary()

    model.fit(frame_seq_dataset, epochs=1)

if __name__ == '__main__':
	main(args)