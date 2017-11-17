import csv
import numpy as np
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras import callbacks
import sys
inputfile = sys.argv[1]
outputfile = sys.argv[2]
# import pandas as pd
# import os
# path = os.environ.get('GRAPE_DATASET_DIR')
# path = ".."
#############
#	config  #
#############
validation = 0.2
batch_size = 300
epochs = 300
def build_model():

    input_img = Input(shape=(48, 48, 1))
    
    block1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
    block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
    block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
    block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)
    block1 = Dropout(0.4)(block1)

    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)
    block2 = Dropout(0.2)(block2)

    block3 = Conv2D(64, (3, 3), activation='relu')(block2)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
    block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)
    block3 = Dropout(0.3)(block3)

    block4 = Conv2D(128, (3, 3), activation='relu')(block3)
    block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)
    # block4 = Flatten()(block4)
    block4 = Dropout(0.35)(block4)       #0.3改0.4

    block5 = Conv2D(128, (3, 3), activation='relu')(block4)
    block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
    block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
    block5 = Dropout(0.2)(block5)       # added

    block5 = Flatten()(block5)

    fc1 = Dense(256, activation='relu')(block5)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(256, activation='relu')(fc1)
    fc2 = Dropout(0.3)(fc2) #origin 0.5

    ## added 
    fc3 = Dense(128, activation='relu')(fc2)
    fc3 = Dropout(0.4)(fc3)

    fc4 = Dense(128,activation='relu')(fc3)
    fc5 = Dropout(0.2)(fc4)
    
    predict = Dense(7)(fc5)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=1e-3)
    opt = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

trainx = []
trainy =[]
testx = []

cnt = 0
model = build_model()
model.load_weights('./report.hdf5', by_name=False)   ########## config
print("finishing loading model")

# with open(path+'/data/test.csv') as csvfile:
with open(inputfile) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # if cnt < 10:
        #     trainx.append(row['feature'].strip().split(' '))
        # else:
        #     break
        trainx.append(row['feature'].strip().split(' '))
        cnt += 1

trainx = np.array(trainx)
trainx = trainx.astype('int')
trainx = (trainx - trainx.mean())/trainx.std()
trainx = trainx.reshape(cnt,48,48,1)

print("testx shape = ",trainx.shape)

ans = model.predict(trainx)

#######################
#   write outputfile  #
#######################
output = ans.argmax(axis=-1)
# fw = open("output/test1.csv",'w')
fw = open(outputfile,'w')
fw.write('id,label\n')
for i in range(len(output)):
    fw.write('{},{}\n'.format(i,output[i]))
