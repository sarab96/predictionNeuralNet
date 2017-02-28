
import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend when there is no DISPLAY or GUI
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from caffe.proto import caffe_pb2

caffe.set_mode_gpu()


# Read model architecture and trained model's weights
net = caffe.Net('cnn_deploy.prototxt',
                caffe.TEST, weights='lstm_iter_37000.caffemodel')

plt.ioff() # turn of interactive plotting mode
# plt.rcParams['figure.figsize'] = (32, 32)
plt.rcParams['image.interpolation'] = 'nearest'

# set size & DPI=100 for 64x64 figure; dpi=100 is selected to support most screen resolutions
# plt.figure(figsize=(0.064, 0.064), dpi=100)  # size is in inches!
# save image in dpi=1000 to get image of size 64x64 pixels
# plt.savefig('myfig.png', dpi=1000)

# plt.title("Training mean image")
# plt.plot(mean_array_forImg.flat)
# plt.axis('off')
# plt.savefig('trainingMeanImage.png')
# plt.close()

'''
Making predicitions
'''

fPredict = open('outputPredictions.txt', 'a')
fPredict.write('=============START===============\n')

# Making predictions
test_ids = []
preds = []
#PowBins,D,Hour,Minute
Tdata = np.genfromtxt('PowOnSineWaveValues.csv', dtype=[('pow','float'), ('day','float'), ('hr','float'), ('min','float')], delimiter=",", skip_header=1)

TdataLen = len(Tdata)
numTimeSteps = 16384 # As need input to conv layer as 128 x 128 with just power input
predictedSteps = 120 # Sarab: Feeding same number of steps into LSTM


i = 33
X = np.zeros((1, numTimeSteps, 1, 1), dtype=np.float32)
for j in range(0, numTimeSteps):
  a = np.asarray(list(Tdata[i+j]))
  #a = np.asarray(Tdata[j])
  X[0, j, 0, 0] = a[0]


expectedPower = np.asarray([])
#p2 = np.asarray([])
#X = np.zeros((numTimeSteps, 2, 3), dtype=np.float32)
#Above resulting in shape of: Top shape: 1 20 2 3 (120) inside Caffe
for j in range(0, predictedSteps):
  a = np.asarray(list(Tdata[i+j+numTimeSteps]))
  #a = np.asarray(Tdata[j])
  expectedPower = np.append(expectedPower, a[0])

minuteSteps = np.arange(predictedSteps)


print('Expected Predicted Power is: ' + str(expectedPower))
print('====================')
plt.plot(minuteSteps, expectedPower, color='blue')
plt.savefig('outputExpectedPowerPlot.png')
#plt.clf()

C = np.ones((predictedSteps, 1), dtype=np.float32)
C[0,0] = 0


net.blobs['data'].data[...] = X
net.blobs['clip'].data[...] = C
out = net.forward()

predictedPower = np.asarray([])
for j in range(0, predictedSteps):
  predictedPower = np.append(predictedPower, out['data_future2'][j][0])

predictedCNNPower = np.asarray([])
for j in range(0, predictedSteps):
  predictedCNNPower = np.append(predictedCNNPower, net.blobs['cnn_data_out_bins'].data[j][0])


print('Predicted Power is: ' + str(predictedPower))
print('====================')
plt.plot(minuteSteps, predictedPower, color='red')
plt.savefig('outputPredictedFinalPowerPlot.png')

plt.plot(minuteSteps, predictedCNNPower, color='green')
plt.savefig('outputPredictedAllPowerPlot.png')

plt.clf()



fPredict.close()




