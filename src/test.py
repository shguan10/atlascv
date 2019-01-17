import numpy as np
import keras
import keras.backend as K
import keras.models as KM
import keras.layers as KL
import keras.engine as KE
from keras import optimizers

def main():
  model = KM.Sequential()
  model.add(KL.Dense(input_dim=10,units=10,kernel_initializer='glorot_uniform',activation='tanh'))
  model.add(KL.Dense(units=10,kernel_initializer='glorot_uniform',activation='tanh'))
  model.add(KL.Dense(units=1,kernel_initializer='glorot_uniform',activation='softmax'))
  x_train = np.random.random((100,10))
  y_train = np.random.random((100,1))

  model.compile(optimizer = 'sgd', loss = 'mean_squared_error',
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