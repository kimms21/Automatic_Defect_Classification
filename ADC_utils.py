
# coding: utf-8

# In[ ]:

import numpy as np
import os
import cv2

from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from random import shuffle


# In[ ]:

# 이미지 전처리 (resize, load binary or rgb)
def getPreProcImgs(path, names, NUM_OF_COLOR_CHANNEL, input_width, input_height):
    ret_list = []
    for r_img in names:
        if NUM_OF_COLOR_CHANNEL==3:
            ttt = cv2.imread(path+r_img)
        else:
            ttt = cv2.imread(path+r_img, 0)
        if ttt is None:
            continue
        equ = cv2.resize(ttt,(input_width, input_height), interpolation=cv2.INTER_AREA)
        ret_list.append(equ)
    return ret_list


# In[ ]:

# path의 이미지 전체 호출
def loadImageSet(path, NUM_OF_COLOR_CHANNEL, input_width, input_height):
    names = os.listdir(path)
    return getPreProcImgs(path, names, NUM_OF_COLOR_CHANNEL, input_width, input_height)


# In[1]:

# path의 이미지 train_ratio:1-train_ration로 train/test set 랜덤하게 생성
def loadTrainTestSet(path, train_ratio, NUM_OF_COLOR_CHANNEL, input_width, input_height):

    names = os.listdir(path)
    n = len(names)
    names = np.array(names)
    
    indices = np.arange(n)
    np.random.shuffle(indices)

    indices_out = indices[int(n * train_ratio):n+1]
    indices = indices[0:int(n * train_ratio)]
    
    return getPreProcImgs(path, names[indices], NUM_OF_COLOR_CHANNEL, input_width, input_height), getPreProcImgs(path, names[indices_out], NUM_OF_COLOR_CHANNEL, input_width, input_height)


# In[ ]:

# 흑백 이미지 rgb 채널로 변경
def gray_to_rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)


# In[ ]:

# Class label을 onehot encoding
def onehotGen(defect, size):
    ret_arr = []
    arr = []
    if defect == 'GA':
        arr = [1., 0., 0., 0.]
    elif defect == 'IP':
        arr = [0., 1., 0., 0.]
    elif defect == 'OP':
        arr = [0., 0., 1., 0.]
    elif defect == 'SA':
        arr = [0., 0., 0., 1.]
#     elif defect == 'SA':
#         arr = [0., 0., 0., 0., 1.]
        
    for i in range(size):
        ret_arr.append(arr)
        
    return ret_arr


# In[ ]:
# defect label 생성
def get_data_labels(defect_lens, defect_names):
    label_list = []
    
    for idx in range(len(defect_names)):
        label_list.append(onehotGen(defect_names[idx], defect_lens[idx]))
    
    label_list = np.array(label_list)
    
    return label_list


# In[ ]:

# VGG-16 기본 Model 호출
def get_normVGGNET16(input_shape=(256, 256, 3), stride=2, init_filter_size=(7,7)):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    # (input_width, input_height, NUM_OF_COLOR_CHANNEL)
#     model.add(Conv2D(64, (7, 7), activation='relu', input_shape=input_shape, strides=stride ))
    model.add(Conv2D(64, init_filter_size, activation='relu', input_shape=input_shape, strides=stride))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4, activation='softmax', name='preds'))
    
    return model


# In[ ]:
def get_random_dataset_xy_single(path, defect_name, train_ratio, NUM_OF_COLOR_CHANNEL, input_width, input_height):
    
    ga_train, ga_test = loadTrainTestSet(path, train_ratio, NUM_OF_COLOR_CHANNEL, input_width, input_height)
    ga_train_y = onehotGen(defect_name,len(ga_train))
    ga_test_y = onehotGen(defect_name,len(ga_test))
   
    train_x = []
    train_y = []

    test_x = []
    test_y = []
    
    if NUM_OF_COLOR_CHANNEL == 1:
        for img in ga_train:
            train_x.append(np.expand_dims(img,-1))
    else:
        for img in ga_train:
            train_x.append(img)

    for label in ga_train_y:
        train_y.append(label)

    # for test
    if NUM_OF_COLOR_CHANNEL == 1:
        for img in ga_test:
            test_x.append(np.expand_dims(img,-1))
    else: 
        for img in ga_test:
            test_x.append(img)

    for label in ga_test_y:
        test_y.append(label)
        
    X_train = np.zeros((len(train_x), input_width, input_height, NUM_OF_COLOR_CHANNEL))
    for idx, x in enumerate(train_x):
        X_train[idx] = x
        
    X_test = np.zeros((len(test_x), input_width, input_height, NUM_OF_COLOR_CHANNEL))
    for idx, x in enumerate(test_x):
        X_test[idx] = x
        
    train_y = np.array(train_y)
    test_y = np.array(test_y)
        
    return X_train, train_y, X_test, test_y


# 모델에 사용 가능한 train/test set 생성
def get_random_dataset_xy(paths, train_ratio, NUM_OF_COLOR_CHANNEL, input_width, input_height):
    
    ga_train, ga_test = loadTrainTestSet(paths[0], train_ratio, NUM_OF_COLOR_CHANNEL, input_width, input_height)
    ip_train, ip_test = loadTrainTestSet(paths[1], train_ratio, NUM_OF_COLOR_CHANNEL, input_width, input_height)
    op_train, op_test = loadTrainTestSet(paths[2], train_ratio, NUM_OF_COLOR_CHANNEL, input_width, input_height)
    sa_train, sa_test = loadTrainTestSet(paths[3], train_ratio, NUM_OF_COLOR_CHANNEL, input_width, input_height)
  
    ga_train_y = onehotGen('GA',len(ga_train))
    ip_train_y = onehotGen('IP',len(ip_train))
    op_train_y = onehotGen('OP',len(op_train))
    sa_train_y = onehotGen('SA',len(sa_train))

    ga_test_y = onehotGen('GA',len(ga_test))
    ip_test_y = onehotGen('IP',len(ip_test))
    op_test_y = onehotGen('OP',len(op_test))
    sa_test_y = onehotGen('SA',len(sa_test))
   
    train_x = []
    train_y = []

    test_x = []
    test_y = []
    
    if NUM_OF_COLOR_CHANNEL == 1:
        for img in ga_train:
            train_x.append(np.expand_dims(img,-1))
        for img in ip_train:
            train_x.append(np.expand_dims(img,-1))
        for img in op_train:
            train_x.append(np.expand_dims(img,-1))
        for img in sa_train:
            train_x.append(np.expand_dims(img,-1))
    else:
        for img in ga_train:
            train_x.append(img)
        for img in ip_train:
            train_x.append(img)
        for img in op_train:
            train_x.append(img)
        for img in sa_train:
            train_x.append(img)

    for label in ga_train_y:
        train_y.append(label)
    for label in ip_train_y:
        train_y.append(label)
    for label in op_train_y:
        train_y.append(label)
    for label in sa_train_y:
        train_y.append(label)

    # for test
    if NUM_OF_COLOR_CHANNEL == 1:
        for img in ga_test:
            test_x.append(np.expand_dims(img,-1))
        for img in ip_test:
            test_x.append(np.expand_dims(img,-1))
        for img in op_test:
            test_x.append(np.expand_dims(img,-1))
        for img in sa_test:
            test_x.append(np.expand_dims(img,-1))

    else: 
        for img in ga_test:
            test_x.append(img)
        for img in ip_test:
            test_x.append(img)
        for img in op_test:
            test_x.append(img)
        for img in sa_test:
            test_x.append(img)

    for label in ga_test_y:
        test_y.append(label)
    for label in ip_test_y:
        test_y.append(label)
    for label in op_test_y:
        test_y.append(label)
    for label in sa_test_y:
        test_y.append(label)
        
    X_train = np.zeros((len(train_x), input_height, input_width, NUM_OF_COLOR_CHANNEL))
    for idx, x in enumerate(train_x):
        X_train[idx] = x
        
    X_test = np.zeros((len(test_x), input_height, input_width, NUM_OF_COLOR_CHANNEL))
    for idx, x in enumerate(test_x):
        X_test[idx] = x
        
    train_y = np.array(train_y)
    test_y = np.array(test_y)
        
    return X_train, train_y, X_test, test_y


# In[ ]:

# 모델에 사용 가능한 data set 생성
def get_test_dataset_xy(paths, NUM_OF_COLOR_CHANNEL, input_width, input_height):
    
    ga_test = loadImageSet(paths[0], NUM_OF_COLOR_CHANNEL, input_width, input_height)
    ip_test = loadImageSet(paths[1], NUM_OF_COLOR_CHANNEL, input_width, input_height)
    op_test = loadImageSet(paths[2], NUM_OF_COLOR_CHANNEL, input_width, input_height)
    sa_test = loadImageSet(paths[3], NUM_OF_COLOR_CHANNEL, input_width, input_height)

    ga_test_y = onehotGen('GA',len(ga_test))
    ip_test_y = onehotGen('IP',len(ip_test))
    op_test_y = onehotGen('OP',len(op_test))
    sa_test_y = onehotGen('SA',len(sa_test))
    
    test_x = []
    test_y = []

    if NUM_OF_COLOR_CHANNEL == 1:
        for img in ga_test:
            test_x.append(np.expand_dims(img,-1))
        for img in ip_test:
            test_x.append(np.expand_dims(img,-1))
        for img in op_test:
            test_x.append(np.expand_dims(img,-1))
        for img in sa_test:
            test_x.append(np.expand_dims(img,-1))

    else: 
        for img in ga_test:
            test_x.append(img)
        for img in ip_test:
            test_x.append(img)
        for img in op_test:
            test_x.append(img)
        for img in sa_test:
            test_x.append(img,)

    for label in ga_test_y:
        test_y.append(label)
    for label in ip_test_y:
        test_y.append(label)
    for label in op_test_y:
        test_y.append(label)
    for label in sa_test_y:
        test_y.append(label)
        
    X_test = np.zeros((len(test_x), input_height, input_width, NUM_OF_COLOR_CHANNEL))
    for idx, x in enumerate(test_x):
        X_test[idx] = x
        
    test_y = np.array(test_y)
    X_test = X_test/255.0
        
    return X_test, test_y


# 모델에 사용 가능한 data set 생성
def get_dataset_uint(paths, NUM_OF_COLOR_CHANNEL, input_width, input_height):
    
    ga_test = loadImageSet(paths[0], NUM_OF_COLOR_CHANNEL, input_width, input_height)
    ip_test = loadImageSet(paths[1], NUM_OF_COLOR_CHANNEL, input_width, input_height)
    op_test = loadImageSet(paths[2], NUM_OF_COLOR_CHANNEL, input_width, input_height)
    sa_test = loadImageSet(paths[3], NUM_OF_COLOR_CHANNEL, input_width, input_height)

    ga_test_y = onehotGen('GA',len(ga_test))
    ip_test_y = onehotGen('IP',len(ip_test))
    op_test_y = onehotGen('OP',len(op_test))
    sa_test_y = onehotGen('SA',len(sa_test))
    
    test_x = []
    test_y = []

    if NUM_OF_COLOR_CHANNEL == 1:
        for img in ga_test:
            test_x.append(np.expand_dims(img,-1))
        for img in ip_test:
            test_x.append(np.expand_dims(img,-1))
        for img in op_test:
            test_x.append(np.expand_dims(img,-1))
        for img in sa_test:
            test_x.append(np.expand_dims(img,-1))

    else: 
        for img in ga_test:
            test_x.append(img)
        for img in ip_test:
            test_x.append(img)
        for img in op_test:
            test_x.append(img)
        for img in sa_test:
            test_x.append(img,)

    for label in ga_test_y:
        test_y.append(label)
    for label in ip_test_y:
        test_y.append(label)
    for label in op_test_y:
        test_y.append(label)
    for label in sa_test_y:
        test_y.append(label)
        
    X_test = np.zeros((len(test_x), input_height, input_width, NUM_OF_COLOR_CHANNEL))
    for idx, x in enumerate(test_x):
        X_test[idx] = x
        
    test_y = np.array(test_y)
    
    X_test = np.array(X_test, dtype='uint8')
    test_y = np.array(test_y, dtype='int8')
        
    return X_test, test_y



# In[ ]:

# Model Accuracy 계산
def getModelAccuracy(pred_result, label):
    pred_y_list = []
    for i in range(len(pred_result)):
        if pred_result[i] == 0:
            pred_y_list.append([1., 0., 0., 0.])
        elif pred_result[i] == 1:
            pred_y_list.append([0., 1., 0., 0.])
        elif pred_result[i] == 2:
            pred_y_list.append([0., 0., 1., 0.])
        elif pred_result[i] == 3:
            pred_y_list.append([0., 0., 0., 1.])
    pred_y_list = np.array(pred_y_list)
    
    same = 0
    diff = 0

    for i in range(len(pred_y_list)):
        if np.array_equal(pred_y_list[i], label[i]):
            same += 1
        else:
            diff += 1
    
    print("Accuracy : " + str( round(same/(same+diff)*100,2)) +"%" )


# In[ ]:



