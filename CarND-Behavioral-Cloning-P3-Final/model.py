import csv
import cv2
import argparse
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.optimizers import Adam
import pylab as plt

def load_data():
    samples = []
    with open(args.base_dir + args.driving_file_log) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

def generator(samples):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, args.batch_size):
            batch_samples = samples[offset:offset+args.batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = args.base_dir + args.image_dir+batch_sample[i].split('/')[-1]
                    current_image = cv2.imread(name)
                    #print(name)
                    current_angle = float(batch_sample[3])
                    images.append(current_image)
                    angles.append(current_angle)

            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement * -1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            #print(X_train[0].shape)
            #print(X_train.shape, y_train.shape)
            yield shuffle(X_train, y_train)

def compile_run(train_samples, validation_samples):
    # compile and train the model using the generator function
    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples)
    ch, row, col = 3, 160, 320

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row,col,ch)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(lr=args.lrate))
    history_object = model.fit_generator(train_generator, samples_per_epoch=
        len(train_samples), validation_data=validation_generator,
        nb_val_samples=len(validation_samples),verbose=1,nb_epoch=args.epoch)
    model.save(args.destfile)

    #### print the keys contained in the history object
    print(history_object.history.keys())
    #### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model to predict steering angles')
    parser.add_argument('--base_dir', default='./data', help='Base directory for input files')
    parser.add_argument('--driving_file_log', default='/driving_log.csv', help='Driving log file')
    parser.add_argument('--image_dir', default='/IMG/', help='Training images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epoch', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lrate', type=float, default=.0001, help='Learning rate')
    parser.add_argument('--destfile', type=str, default='model.h5', help='Persisted model file ')
    args = parser.parse_args()
    print(args.batch_size, args.epoch, args.destfile, args.driving_file_log, args.image_dir)

    samples = load_data()
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    compile_run(train_samples, validation_samples)
