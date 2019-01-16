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
    mask = KL.Conv2DTranspose(2,(3,3),
                           activation='softmax',
                           strides=(32,32),
                           padding='same',
                           name='end_32trans',
                           kernel_initializer="glorot_uniform")(x)

    self.model = KM.Model(inputs=vgg.input,outputs=mask)

  def compile(self):
    self.model.compile(optimizer='sgd',loss='mean_squared_error',metrics='accuracy')

  def train(self,traindata=None,testdata=None):
    self.traindata = traindata
    self.testdata = testdata
    savecb = keras.callbacks.ModelCheckpoint(filepath='chkpnts/fcn_vacc{val_acc:.3f}.h5', 
                                           monitor='val_acc', 
                                           verbose=1, 
                                           save_best_only=True)




def read_img(img_path,resize=True):
  img = image.load_img(img_path, target_size=(666,375)) if resize else image.load_img(img_path)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  return preprocess_input(x)

from os import listdir as ls
def get_imgs(imgdir='/home/xinyu/school/build18/atlascv/data_road/training/image_2'):
  x = [image.img_to_array(image.load_img(imgdir+"/"+img)) for img in ls(imgdir)]
  x = np.array(x)
  assert len(x.shape)==4, "shape mismatch"
  return preprocess_input(x)

def get_masks(imgdir='/home/xinyu/school/build18/atlascv/data_road/training/gt_image_2'):
  def img2mask(img):
    return np.array([[c[2]>12 for c in r] for r in img])
  x = [np.expand_dims(image.img_to_array(image.load_img(imgdir+"/"+img)),axis=0) for img in ls(imgdir)]
  import pdb
  pdb.set_trace()
  x = np.array(x)
  # ishape = x[0].shape
  # newshape = (x.shape[0],ishape[0],ishape[1],ishape[2])
  print(x.shape)
  print(type(x[0]))
  # x = x.reshape(newshape)
  assert len(x.shape)==4, "shape mismatch"
  return np.vectorize(img2mask)(x)

def test():
  testimg = np.arange(24).reshape(2,3,4)

  genTF = lambda cc: cc[2]>12

  f = np.vectorize(np.vectorize(genTF))
  f(testimg)
  import pdb
  pdb.set_trace()

if __name__ == '__main__':
  get_masks()

# cat = read_img('cat.jpg')

# fcn = FCN()
# fcn.build()

# import timeit
# print(timeit.timeit('fcn.model.predict(cat)',globals=globals(),number=1000))
