# libs for model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import numpy as np

#tensorboard
NAME ="Model_NAME_{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir=r"D:\MLD\logfile\{}".format(NAME))
#load pickle
pickle_in = open(r"D:\MLD\pickle_save\data.pickle","rb")
data = pickle.load(pickle_in)
pickle_in = open(r"D:\MLD\pickle_save\label.pickle","rb")
label = pickle.load(pickle_in)
#transform to numpy array
data = np.array(data).reshape(-1,50,50,3)

#label = np.array(label)
label = tf.keras.utils.to_categorical(label)

#normalize the data to [0,1]
data = data / 255.0

#build model
model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=data.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(1,1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
#model.add(Dense(64))
#model.add(Activation("relu"))
#model.add(Dropout(0.2))

model.add(Dense(20))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",
            optimizer="adam",
            metrics=['accuracy'])

model.fit(data,label,epochs=10,batch_size=32,validation_split=0.2,callbacks=[tensorboard])
