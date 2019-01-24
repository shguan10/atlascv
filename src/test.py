import numpy as np
import keras
import keras.backend as K
import keras.models as KM
import keras.layers as KL
import keras.engine as KE
from keras import optimizers

import tensorflow as tf

def main():
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  
  fashion_mnist = keras.datasets.fashion_mnist

  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

  print(x_train.shape)
  print(x_test.shape)

  print(y_train.shape)
  print(y_test.shape)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')

  x_train = x_train / 255.
  x_test /= 255.

  model.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'])

  save_cb = keras.callbacks.ModelCheckpoint("fake/weights.{epoch:02d}-{val_acc:.2f}.hdf5",
                                            monitor="val_acc", period=5)

  history = model.fit(x_train, y_train, batch_size = 10, epochs = 100, verbose = 1, callbacks=[save_cb],
                          shuffle=1, validation_data = (x_train, y_train))
  import matplotlib.pyplot as plt
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

if __name__ == '__main__':
  main()