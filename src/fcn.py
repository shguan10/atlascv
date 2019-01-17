from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import keras
import keras.backend as K
import keras.models as KM
import keras.layers as KL
import keras.engine as KE
from keras import optimizers

import numpy as np

class FCN:
  def __init__(self):
    pass
  def build(self):
    vgg = VGG16(weights='imagenet', include_top=False,variable_inp_size = True)

    x = KL.Conv2D(2, (1, 1),
                  activation='relu',
                  padding='same',
                  name='end_conv1')(vgg.output)
    mask = KL.Conv2DTranspose(1,(3,3),
                           activation='softmax',
                           strides=(32,32),
                           padding='same',
                           name='end_32trans',
                           kernel_initializer="glorot_uniform")(x)

    self.model = KM.Model(inputs=vgg.input,outputs=mask)
    self.model.summary()
  def compile(self):
    self.model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])

  def train(self,traindata=None,testdata=None):
    x_train,y_train = traindata
    x_test,y_test = testdata
    savecb = keras.callbacks.ModelCheckpoint(filepath='chkpnts/fcn_acc{acc:.3f}.h5', 
                                           monitor='acc', 
                                           verbose=1, 
                                           period = 5,
                                           save_best_only=True)
    history = self.model.fit(x_train, y_train, batch_size = 10, epochs = 100, verbose = 1, callbacks=[],
                            shuffle=1, validation_data = (x_test, y_test))
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

def read_img(img_path,resize=True):
  img = image.load_img(img_path, target_size=(666,375)) if resize else image.load_img(img_path)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  return preprocess_input(x)

from os import listdir as ls
def get_imgs(imgdir='/home/xinyu/school/atlascv/data_road/training/image_2/cropped'):
  with open("imgs.txt",'r') as f:
    imgs = [x[:-1] for x in f]
  x = [image.img_to_array(image.load_img(imgdir+"/"+img)) for img in imgs]
  x = np.array(x)
  assert len(x.shape)==4, "shape mismatch"
  return preprocess_input(x)

def get_masks(imgdir='/home/xinyu/school/atlascv/data_road/training/masks/cropped'):
  with open("masks.txt",'r') as f:
    masks = [x[:-1] for x in f]
  x = [image.img_to_array(image.load_img(imgdir+"/"+img)) for img in masks]
  x = np.array(x)
  print(x.shape)
  assert len(x.shape)==4, "shape mismatch"
  return np.array([[[c[2]>12 for c in r] for r in img] for img in x])

if __name__ == '__main__':
  import pickle as pk
  masks = pk.load(open('masks.pk','rb'))
  masks = np.array([[[int(c) for c in r]for r in img]for img in masks])
  masks = np.expand_dims(masks,axis=3)
  imgs = pk.load(open('imgs.pk','rb'))
  assert masks.shape[0]==289, 'shape mismatch'

  TRAINS = 231
  TESTS = 289-TRAINS #80/20 split
  masks_train = masks[:289]
  imgs_train = imgs[:289]
  masks_test = masks[289:]
  imgs_test = imgs[289:]
  fcn = FCN()
  fcn.build()
  fcn.compile()
  fcn.train(traindata=(imgs_train,masks_train),testdata=(imgs_test,masks_test))

# cat = read_img('cat.jpg')

# fcn = FCN()
# fcn.build()

# import timeit
# print(timeit.timeit('fcn.model.predict(cat)',globals=globals(),number=1000))
