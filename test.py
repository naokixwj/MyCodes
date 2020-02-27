'''
0-airplane
1-automobile
2-bird
3-cat
5-dog
6-frog
7-horse
8-ship
9-truck
'''
#C:\Users\xwjtb\Desktop\tf_ObjectDetection\protoc\bin\protoc C:\Users\xwjtb\Desktop\tf_ObjectDetection\models\research\object_detection\protos\*.proto --python_out=.
import time


Convs = [1,2,3]
Size = [32,64,128]
Dense = [0,1,2]
for size in Size:
    for conv in Convs:
        for dense in Dense:
            NAME = "CONV_{}_DENSE_{}_SIZE_{}_{}".format(conv,dense,size,int(time.time()))
            print(NAME)
            for i in range(conv):
                print(i)
            for i in range(dense):
                print(i)