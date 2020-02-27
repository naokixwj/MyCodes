## Imports
import os
import sys
import random
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, concatenate, Dropout,\
                                    Lambda, Conv2DTranspose, Add
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
'''
cc_codes = [
    '0110',
    '0120',
    '0131',
    '0132',
    '0133',
    '0140',
    '0170',
    '0180',
    '0191',
    '0311',
    '0312',
    '0313',
    '0321',
    '0340',
    '0360',
    '0370',
    '0391',
    '03A2',
    '03A4',
    '03A9',
    '0511',
    '0521',
    '0530',
    '0541',
    '0542',
    '0543',
    '0544',
    '0550',
    '0601',
    '0610',
    '0710',
    '0711',
    '0712',
    '0713',
    '0715',
    '0716',
    '0717',
    '0718',
    '0719',
    '0721',
    '0750',
    '0760',
    '0761',
    '0762',
    '0769',
    '0770',
    '0790',
    '0814',
    '0819',
    '0822',
    '0829',
    '0831',
    '0832',
    '0833',
    '0839',
    '0890',
    '0920',
    '0940',
    '1001'
]
'''
## Seeding 
seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed
cc_codes = [
    'circle',
    'star',
    'square',
    'triangle'
]
class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=128):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        
    def __load__(self, id_name):
        ## Path
        image_path = os.path.join(self.path, id_name, "image", "IMG"+id_name.split('_')[0]) + ".png"
        mask_path = os.path.join(self.path, id_name, "mask\\")
        all_masks = os.listdir(mask_path)
        
        ## Reading Image
        image = cv2.imread(image_path, 1)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        #mask = cv2.imread(mask_path, 0)
        #mask = cv2.resize(mask, (self.image_size, self.image_size))

        channels = cc_codes[:]
        ## Reading Masks background = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)
        background = np.ones((self.image_size, self.image_size),dtype=np.float32)
        background = background * 255
        for name in all_masks:
            if name.endswith('png'):
                img_cc = name[4:len(name)-4]
                _mask_path = mask_path + name
                _mask_image = cv2.imread(_mask_path, 0)
                _mask_image = cv2.resize(_mask_image, (self.image_size, self.image_size)) #128x128
                #_mask_image = np.expand_dims(_mask_image, axis=-1)
                for idx,cckey in enumerate(channels):
                    if cckey == img_cc:
                        channels[idx] = _mask_image
                background = background - _mask_image #反色处理(INV)
                #background = np.maximum(background, _mask_image)
        #result = background[:, :]
        #plt.imshow(result)
        #plt.show()
        for idx,cckey in enumerate(channels):
            if len(channels[idx]) == len(cc_codes[idx]):
                channels[idx] = np.zeros((self.image_size, self.image_size),dtype=np.float32)
            else:
                channels[idx] = cckey / (idx + 2)
            
        channels.append(background)
        #for c in channels:
            #result = c[:, :]
            #plt.imshow(result)
            #plt.show()
        mask = np.stack(channels,axis=2)
        ## Normalizaing 
        image = image/255.0
        mask = mask/255.0
        
        return image, mask
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask  = []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
            
        image = np.array(image)
        mask  = np.array(mask)
        
        return image, mask
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

image_size = 256
train_path = r"G:\MLD\Semantic-Shapes-master\annotated_png"


## Training Ids
train_ids = next(os.walk(train_path))[1]
random.shuffle(train_ids)
#
batch_size = 8
epochs = 100


## Validation Data Size
val_data_size = int(len(train_ids)*0.1)

valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

gen = DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)
x, y = gen.__getitem__(0)
print(x.shape, y.shape)

def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="elu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="elu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="elu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="elu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="elu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="elu")(c)
    return c

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def UNet0():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    outputs = keras.layers.Conv2D(5, (1, 1), padding="same", activation="softmax")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

def UNet():
    b = 4
    inputs = keras.layers.Input((image_size, image_size, 3))

    s = Lambda(lambda x: preprocess_input(x)) (inputs)
    
    c1 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(2**(b+4), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(2**(b+4), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(2**(b+3), (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(2**(b+2), (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(2**(b+1), (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(2**b, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = keras.layers.Conv2D(5, (1, 1), padding="same", activation="softmax")(c9)
    
    model = keras.models.Model(inputs, outputs)
    return model

model = UNet0()
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["acc"])
model.summary()

train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)

train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size

model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, 
                    epochs=epochs)
## Save the Weights
model.save_weights("G:\MLD\SethTask.h5")