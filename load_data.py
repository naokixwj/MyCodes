import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATA_DIR = r"D:\MLD\cifar10-pngs-in-folders\cifar10\cifar10\train"
Classes = os.listdir(DATA_DIR)
Images = []
Labels = []
Images_Labels = []
for index,_class in enumerate(Classes):
    print("{}-{}".format(index,_class))
    class_path = os.path.join(DATA_DIR,_class)
    files = os.listdir(class_path)
    for img in files:
        try:
            img_array = cv2.imread(os.path.join(class_path,img),cv2.IMREAD_COLOR)
            new_array = cv2.resize(img_array,(50,50))
            Images_Labels.append([new_array,index])
        except Exception as e:
            pass
random.shuffle(Images_Labels)
for x,y in Images_Labels:
    Images.append(x)
    Labels.append(y)
for sample in Images_Labels[:10]:
    print(sample[1])
pickle_out=open(r"D:\MLD\pickle_save\cifar-data.pickle","wb")
pickle.dump(Images,pickle_out)
pickle_out.close()
pickle_out=open(r"D:\MLD\pickle_save\cifar-label.pickle","wb")
pickle.dump(Labels,pickle_out)
pickle_out.close()
