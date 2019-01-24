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
import matplotlib.pyplot as plt
import pdb

class FCN:
  def __init__(self):
    pass
  def build(self):
    vgg = VGG16(weights='imagenet', include_top=False,variable_inp_size = True)

    x = KL.Conv2D(2, (1, 1),
                  activation='relu',
                  padding='same',
                  name='end_conv',
                  kernel_initializer="glorot_uniform")(vgg.output)
    x = KL.Conv2DTranspose(1,(3,3),
                           activation='sigmoid',
                           strides=(32,32),
                           padding='same',
                           name='end_32trans',
                           kernel_initializer="glorot_uniform")(x)
    y = KL.Conv2D(2, (1, 1),
                  activation='relu',
                  padding='same',
                  name='end_conv4',
                  kernel_initializer="glorot_uniform")(vgg.get_layer('block4_pool').output)
    y = KL.Conv2DTranspose(1,(3,3),
                           activation='sigmoid',
                           strides=(16,16),
                           padding='same',
                           name='end_16trans',
                           kernel_initializer="glorot_uniform")(y)
    z = KL.Conv2D(2, (1, 1),
                  activation='relu',
                  padding='same',
                  name='end_conv3',
                  kernel_initializer="glorot_uniform")(vgg.get_layer('block3_pool').output)
    z = KL.Conv2DTranspose(1,(3,3),
                           activation='sigmoid',
                           strides=(8,8),
                           padding='same',
                           name='end_8trans',
                           kernel_initializer="glorot_uniform")(z)
    x = KL.Concatenate()([x,y,z])
    x = KL.Conv2D(1, (1, 1),
                  activation='sigmoid',
                  padding='same',
                  name='end',
                  kernel_initializer="glorot_uniform")(x)
    
    self.model = KM.Model(inputs=vgg.input,outputs=x)
    self.model.summary()
  def compile(self):
    sgd = optimizers.SGD(lr=.01, decay=1e-6, momentum=0.9, nesterov=True)
    self.model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['binary_accuracy'])

  def loadweights(self,wpath):
    self.model.load_weights(wpath)

  def train(self,traindata=None,testdata=None,wpath=None):
    x_train,y_train = traindata
    x_test,y_test = testdata
    savecb = keras.callbacks.ModelCheckpoint(filepath='chkpnts/fcn_acc{val_binary_accuracy:.3f}.h5', 
                                           monitor='val_binary_accuracy', 
                                           verbose=1, 
                                           period = 10,
                                           save_best_only=True)
    if wpath is not None: self.model.load_weights(wpath)
    history = self.model.fit(x_train, y_train, batch_size = 10, epochs = 1600, verbose = 1, callbacks=[savecb],
                             shuffle=1, validation_data = (x_test, y_test))
    
    # summarize history for accuracy
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
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
  img = image.load_img(img_path, target_size=(384,672)) if resize else image.load_img(img_path)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  return preprocess_input(x)

def show_mask(mask,savepath=None):
  if len(mask.shape) == 4:
    mask = mask[0]
  if len(mask.shape) == 3:
    mask.reshape(mask.shape[:2])

  # img = np.array([[[el,el,el]  for el in r]for r in mask])
  img = np.array([[[el,el,el] for el in r]for r in mask])
  img = img.reshape(img.shape[:3])
  plt.imshow(img)
  plt.show()

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
  x/=255.
  return np.array([[[c[2] for c in r] for r in img] for img in x])

def getdata():
  import pickle as pk
  masks = pk.load(open('masks.pk','rb'))
  imgs = pk.load(open('imgs.pk','rb'))
  assert masks.shape[0]==289, 'shape mismatch'
  TRAINS = 231
  TESTS = 289-TRAINS #80/20 split
  masks_train = masks[:TRAINS]
  imgs_train = imgs[:TRAINS]
  masks_test = masks[TRAINS:]
  imgs_test = imgs[TRAINS:]
  return imgs_train,masks_train,imgs_test,masks_test


def train():
  imgs_train,masks_train,imgs_test,masks_test = getdata()
  fcn = FCN()
  fcn.build()
  fcn.compile()
  fcn.train(traindata=(imgs_train,masks_train),testdata=(imgs_test,masks_test),wpath='chkpnts/fcn_acc0.804.h5')
  return fcn

def getmodel(weights='chkpnts/fcn_acc0.804.h5'):
  fcn = FCN()
  fcn.build()
  if weights is not None: fcn.loadweights(weights)
  return fcn

def apply_mask(image, mask, color, alpha=0.5,colornormalized = True):
    """Apply the given mask to the image.
      code adapted from mrcnn repo
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == True,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * (255 if colornormalized else 1),
                                  image[:, :, c])
    return image

def rescaleimg(img):
  # rescales so every pixel is (0,1)
  pmax = np.max(img)
  pmin = np.min(img)
  avg = (pmax+pmin)/2
  dist = pmax - pmin
  return (img - avg)/dist + 0.5

def infersave():
  fcn = getmodel()
  imgs_train,masks_train,imgs_test,masks_test = getdata()
  # sel_train = np.random.choice(np.arange(len(imgs_train)),10)
  # sel_test = np.random.choice(np.arange(len(imgs_test)),10)
  # imgs_train = np.array([imgs_train[x] for x in sel_train])
  # masks_train = [masks_train[x] for x in sel_train]
  # imgs_test = np.array([imgs_test[x] for x in sel_test])
  # masks_test = [masks_test[x] for x in sel_test]

  masks_pred = [fcn.model.predict(imgs_train[10*x:10*x+10]) for x in range(len(imgs_train)//10)]
  masks_pred = np.array([item for sublist in masks_pred for item in sublist])
  masks_pred = masks_pred > 0.5
  for ind,(img,mask_pred,mask_true) in enumerate(zip(imgs_train,masks_pred,masks_train)):
    mask_pred = mask_pred.reshape(mask_pred.shape[:2])
    mask_true = mask_true.reshape(mask_true.shape[:2])
    img = rescaleimg(img)
    img_p = img.copy()
    img_p = apply_mask(img_p,mask_pred,(0.1999999999999993, 0.0, 1.0))
    plt.imshow(img_p)
    plt.savefig("results/train/mask_pred_train%d.png" % ind)
    img = apply_mask(img,mask_true,(0.1999999999999993, 0.0, 1.0))
    plt.imshow(img)
    plt.savefig("results/train/mask_true_train%d.png" % ind)

  masks_pred = [fcn.model.predict(imgs_test[10*x:10*x+10]) for x in range(len(imgs_test)//10)]
  masks_pred = np.array([item for sublist in masks_pred for item in sublist])
  masks_pred = masks_pred > 0.5
  for ind,(img,mask_pred,mask_true) in enumerate(zip(imgs_test,masks_pred,masks_test)):
    mask_pred = mask_pred.reshape(mask_pred.shape[:2])
    mask_true = mask_true.reshape(mask_true.shape[:2])
    img = rescaleimg(img)
    img_p = img.copy()
    img_p = apply_mask(img_p,mask_pred,(0.1999999999999993, 0.0, 1.0))
    plt.imshow(img_p)
    plt.savefig("results/test/mask_pred_test%d.png" % ind)
    img = apply_mask(img,mask_true,(0.1999999999999993, 0.0, 1.0))
    plt.imshow(img)
    plt.savefig("results/test/mask_true_test%d.png" % ind)

if __name__ == '__main__':
  # infersave()
  from keras.utils import plot_model
  fcn = getmodel()
  plot_model(fcn.model,'model.png')
  # train()

# cat = read_img('cat.jpg')

# fcn = FCN()
# fcn.build()

# import timeit
# print(timeit.timeit('fcn.model.predict(cat)',globals=globals(),number=1000))
