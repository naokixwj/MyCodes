# libs for model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import numpy as np

#tensorboard

#load pickle
pickle_in = open(r"E:\MLD\pickle_save\cifar-data.pickle","rb")
data = pickle.load(pickle_in)
pickle_in = open(r"E:\MLD\pickle_save\cifar-label.pickle","rb")
label = pickle.load(pickle_in)
#transform to numpy array
data = np.array(data).reshape(-1,50,50,3)

#label = np.array(label)
label = tf.keras.utils.to_categorical(label)

#normalize the data to [0,1]
data = data / 255.0

Convs = [3]
Size = [128]
_Dense = [1]

#build model

for size in Size:
    for conv in Convs:
        for dense in _Dense:
            NAME = "CONV_{}_DENSE_{}_SIZE_{}_{}".format(conv,dense,size,int(time.time()))
            tensorboard = TensorBoard(log_dir=r"E:\MLD\logfile\{}".format(NAME))
            model = Sequential()
            model.add(Conv2D(size,(3,3),input_shape=data.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))
            for i in range(conv):
                model.add(Conv2D(size,(3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Flatten())
            for i in range(dense): 
                model.add(Dense(size))
                model.add(Activation("relu"))
                model.add(Dropout(0.2))
            model.add(Dense(10))
            model.add(Activation("softmax"))
            model.compile(loss="categorical_crossentropy",
                        optimizer="rmsprop",
                        metrics=['accuracy'])
            model.fit(data,label,epochs=10,batch_size=32,validation_split=0.2,callbacks=[tensorboard])





