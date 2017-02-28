import os
import glob
import random
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
#matplotlib inline

#import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

npVer = np.version.version

train_lmdb = './train_lmdb'
test_lmdb = './test_lmdb'
trainLabel_lmdb = './trainLabel_lmdb'
testLabel_lmdb = './testLabel_lmdb'
trainClip_lmdb = './trainClip_lmdb'
testClip_lmdb = './testClip_lmdb'


fDebug = open('outputDebug.txt', 'a')
fDebug.write('Numpy version is: ' + str(npVer))
fDebug.write('\n=============START===============\n')
fDebug.write('Deleting old data in DB lmdb\n')

# Delete old lmdb, as creating new one below
os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + test_lmdb)
os.system('rm -rf  ' + trainLabel_lmdb)
os.system('rm -rf  ' + testLabel_lmdb)
os.system('rm -rf  ' + trainClip_lmdb)
os.system('rm -rf  ' + testClip_lmdb)


#my_data = genfromtxt('my_file.csv', delimiter=',')
#PowBins,D,Hour,Minute
Tdata = np.genfromtxt('PowOnSineWaveValues.csv', dtype=[('pow','float'), ('day','float'), ('hr','float'), ('min','float')], delimiter=",", skip_header=1)

#Tdata[0] is: (0.12621318, 0.017044348, 0.01073578, 0.115257832)
#Tdata[1] is: (0.123589, -0.019392899, 0.009858902, 0.110947488)
#Tdata[0][0] is: 0.12621318
#Tdata[0][1] is: 0.017044348

#Sarab: Our Numpy version is: 1.10.4

#fDebug.write('\nTdata[0] is: ' + str(Tdata[0]) + '\n')
#fDebug.write('Tdata[1] is: ' + str(Tdata[1]) + '\n')
#fDebug.write('\nTdata[len-1] is: ' + str(Tdata[TdataLen-1]) + '\n')
#fDebug.write('Tdata[len-2] is: ' + str(Tdata[TdataLen-2]) + '\n')
#Tdata[0] is: (0.12621318, 0.017044348, 0.01073578, 0.115257832)
#Tdata[1] is: (0.123589, -0.019392899, 0.009858902, 0.110947488)
#Tdata[0][0] is: 0.12621318
#Tdata[0][1] is: 0.017044348
#someTuple = (1,2,3,4,...)
#someList  = [1,2,3,4,...] 
#a list of values. Each one of them is numbered, starting from zero - the first one is numbered zero, the second 1, the third 2, etc.
#Tuples are just like lists, but you can't change their values. The values that you give it first up, are the values that you are stuck with for the rest of the program. Again, each value is numbered starting from zero, for easy reference.
#array([1, 2, 3, 4, 5, 6, 7, 8, 9])
#array([[1, 2, 3],
#       [4, 5, 6],
#       [7, 8, 9]])


fDebug.write('Creating train_lmdb\n')

TdataLen = len(Tdata)
numTimeSteps = 16384 # As need input to conv layer as 128 x 128 with just power input
predictedSteps = 120 # Sarab: Feeding same number of steps into LSTM

trainData_db = lmdb.open(train_lmdb, map_size=int(1e12), map_async=True, writemap=True)
trainLabels_db = lmdb.open(trainLabel_lmdb, map_size=int(1e12), map_async=True, writemap=True)
#Using writemap=True causes LMDB to directly write your changes to a file-backed memory mapping, which behaves like a "write-back cache". This means Linux will keep changes in memory until memory is low, at which point, it starts IO to write the dirty pages to disk to satisfy the memory pressure.
#map_async: When writemap=True, use asynchronous flushes to disk. As with sync=False, a system crash can then corrupt the database or lose the last transactions. Calling sync() ensures on-disk database integrity until next commit.

trainData_txn = trainData_db.begin(write=True)
trainLabels_txn = trainLabels_db.begin(write=True)



i = 0
k = 0
#PowBins,D,Hour,Minute
while (i < TdataLen-numTimeSteps-predictedSteps-10):
    #X = np.zeros((numTimeSteps, 2, 3), dtype=np.float32)
    #Above resulting in shape of: Top shape: 1 20 2 3 (120) inside Caffe
    X = np.zeros((numTimeSteps, 1, 1), dtype=np.float32)
    for j in range(0, numTimeSteps):
      a = np.asarray(list(Tdata[i+j]))
      #a = np.asarray(Tdata[j])
      X[j, 0, 0] = a[0]

    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = X.shape[0]
    datum.height = X.shape[1]
    datum.width = X.shape[2]
    datum.float_data.extend(X[:].astype(float).flat)
    key_id = '{:08}'.format(k)
    # The encode is only essential in Python 3
    trainData_txn.put(key_id.encode('ascii'), datum.SerializeToString())

    XL = np.zeros((1, 1, predictedSteps), dtype=np.float32)
    p1 = np.asarray([])
    for j in range(0, predictedSteps):
      a = np.asarray(list(Tdata[i+numTimeSteps+j]))
      #a = np.asarray(Tdata[j])
      p1 = np.append(p1, a[0])

    XL[0,0] = p1
    datumL = caffe.proto.caffe_pb2.Datum()
    datumL.channels = XL.shape[0]
    datumL.height = XL.shape[1]
    datumL.width = XL.shape[2]
    datumL.float_data.extend(XL[:].astype(float).flat)
    trainLabels_txn.put(key_id.encode('ascii'), datumL.SerializeToString())

    k += 1
    i = i + 233

trainData_txn.commit() #Sarab: Do not commit if using "with statement" to get trainData_txn; as that implicitly does commit; doing commit twice may give error.
trainLabels_txn.commit()
trainData_db.close()
trainLabels_db.close()
fDebug.write('trainData last index is: ' + str(k) + '\n')

fDebug.write('\nCreating trainClip_lmdb\n')

in_db = lmdb.open(trainClip_lmdb, map_size=int(1e12))
i = 0
k = 0
with in_db.begin(write=True) as in_txn:
    while (i < TdataLen-numTimeSteps-predictedSteps-10):
        #X = np.zeros((numTimeSteps, 2, 3), dtype=np.float32)
        #Above resulting in shape of: Top shape: 1 20 2 3 (120) inside Caffe
        X = np.ones((predictedSteps, 1, 1), dtype=np.float32)
        X[0,0,0] = 0

        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[0]
        datum.height = X.shape[1]
        datum.width = X.shape[2]
        datum.float_data.extend(X[:].astype(float).flat)
        key_id = '{:08}'.format(k)
        # The encode is only essential in Python 3
        in_txn.put(key_id.encode('ascii'), datum.SerializeToString())
        k += 1
        i = i + 233

in_db.close()
fDebug.write('trainClip_lmdb last index is: ' + str(k) + '\n')



testData_db = lmdb.open(test_lmdb, map_size=int(1e12), map_async=True, writemap=True)
testLabels_db = lmdb.open(testLabel_lmdb, map_size=int(1e12), map_async=True, writemap=True)
#Using writemap=True causes LMDB to directly write your changes to a file-backed memory mapping, which behaves like a "write-back cache". This means Linux will keep changes in memory until memory is low, at which point, it starts IO to write the dirty pages to disk to satisfy the memory pressure.
#map_async: When writemap=True, use asynchronous flushes to disk. As with sync=False, a system crash can then corrupt the database or lose the last transactions. Calling sync() ensures on-disk database integrity until next commit.

testData_txn = testData_db.begin(write=True)
testLabels_txn = testLabels_db.begin(write=True)



i = 0
k = 0
#PowBins,D,Hour,Minute
while (i < TdataLen-numTimeSteps-predictedSteps-10):
    #X = np.zeros((numTimeSteps, 2, 3), dtype=np.float32)
    #Above resulting in shape of: Top shape: 1 20 2 3 (120) inside Caffe
    X = np.zeros((numTimeSteps, 1, 1), dtype=np.float32)
    for j in range(0, numTimeSteps):
      a = np.asarray(list(Tdata[i+j]))
      #a = np.asarray(Tdata[j])
      X[j, 0, 0] = a[0]

    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = X.shape[0]
    datum.height = X.shape[1]
    datum.width = X.shape[2]
    datum.float_data.extend(X[:].astype(float).flat)
    key_id = '{:08}'.format(k)
    # The encode is only essential in Python 3
    testData_txn.put(key_id.encode('ascii'), datum.SerializeToString())

    XL = np.zeros((1, 1, predictedSteps), dtype=np.float32)
    p1 = np.asarray([])
    for j in range(0, predictedSteps):
      a = np.asarray(list(Tdata[i+numTimeSteps+j]))
      #a = np.asarray(Tdata[j])
      p1 = np.append(p1, a[0])

    XL[0,0] = p1
    datumL = caffe.proto.caffe_pb2.Datum()
    datumL.channels = XL.shape[0]
    datumL.height = XL.shape[1]
    datumL.width = XL.shape[2]
    datumL.float_data.extend(XL[:].astype(float).flat)
    testLabels_txn.put(key_id.encode('ascii'), datumL.SerializeToString())

    k += 1
    i = i + 777

testData_txn.commit() #Sarab: Do not commit if using "with statement" to get trainData_txn; as that implicitly does commit; doing commit twice may give error.
testLabels_txn.commit()
testData_db.close()
testLabels_db.close()
fDebug.write('trainData last index is: ' + str(k) + '\n')

fDebug.write('\nCreating testClip_lmdb\n')

in_db = lmdb.open(testClip_lmdb, map_size=int(1e12))
i = 0
k = 0
with in_db.begin(write=True) as in_txn:
    while (i < TdataLen-numTimeSteps-predictedSteps-10):
        #X = np.zeros((numTimeSteps, 2, 3), dtype=np.float32)
        #Above resulting in shape of: Top shape: 1 20 2 3 (120) inside Caffe
        X = np.ones((predictedSteps, 1, 1), dtype=np.float32)
        X[0,0,0] = 0

        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[0]
        datum.height = X.shape[1]
        datum.width = X.shape[2]
        datum.float_data.extend(X[:].astype(float).flat)
        key_id = '{:08}'.format(k)
        # The encode is only essential in Python 3
        in_txn.put(key_id.encode('ascii'), datum.SerializeToString())
        k += 1
        i = i + 777

in_db.close()
fDebug.write('testClip_lmdb last index is: ' + str(k) + '\n')











fDebug.write('\nFinished processing all data\n')

# Use the function cv2.imwrite() to save an image.
# First argument is the file name, second argument is the image you want to save.
# cv2.imwrite('messigray.png',img)
# This will save the image in PNG format in the working directory.

fDebug.close()
