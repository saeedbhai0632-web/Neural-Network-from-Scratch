import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import gzip
import pickle

np.random.seed(0)

# LOAD DATA(done with AI )
import ssl
ctx=ssl.create_default_context()
ctx.check_hostname=False
ctx.verify_mode=ssl.CERT_NONE

import urllib.request
opener=urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
opener.retrieve=urllib.request.urlretrieve
urllib.request.install_opener(opener)

urllib.request.urlretrieve(
    "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz",
    "mnist.pkl.gz"
)
with gzip.open("mnist.pkl.gz", "rb") as f:
    (X_train,y_train),(X_val,y_val),(X_test,y_test)=pickle.load(f,encoding="latin1")

y_train=y_train.astype(int)
y_test=y_test.astype(int)

class layer:
    def __init__(self,inputCnt,neuronCnt):
        self.weights=np.random.randn(inputCnt,neuronCnt)*0.1
        self.biases=np.zeros((1,neuronCnt))

    def process(self,inputs):
        self.input=inputs
        self.output=np.dot(inputs,self.weights)+self.biases

class ReLU:
    def process(self,inputs):
        self.input=inputs
        self.output=np.maximum(inputs,0)

class softmax:
    def process(self,inputs):
        expo=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        sum1=np.sum(expo,axis=1,keepdims=True)
        prob=expo/sum1
        self.output=prob

class LCCE:
    def forward(self,prediction,real):
        predClipped=np.clip(prediction,1e-7,1-1e-7)
        corrConfi=predClipped[range(len(prediction)),real]
        negLog=-np.log(corrConfi)
        return np.mean(negLog)

class backPropL1:
    def deltaCalW(self,prevVals,aL,real):
        batchsize=len(aL)
        delta=aL.copy()
        delta[range(batchsize),real]-=1
        delta=delta/batchsize
        return np.dot(prevVals.T, delta)
    def deltaCalb(self,aL,real):
        batchsize=len(aL)
        delta = aL.copy()
        delta[range(batchsize),real]-=1
        delta = delta / batchsize
        return np.sum(delta, axis=0, keepdims=True)
    def deltaCurr(self,real,aL):
        batchsize=len(aL)
        delta=aL.copy()
        delta[range(batchsize),real]-=1
        delta=delta/batchsize
        return delta

class backProp:
    def deltaCalW(self,prevVals,aL,nextWts,deltaNext):
        delta=np.dot(deltaNext,nextWts.T)
        delta*=(aL.input>0)
        return np.dot(prevVals.T,delta)
    def deltaCalb(self,aL,nextWts,deltaNext):
        delta = np.dot(deltaNext,nextWts.T)
        delta*=(aL.input>0)
        return np.sum(delta,axis=0,keepdims=True)
    def deltaCurr(self,aL,nextWts,deltaNext):
        return np.dot(deltaNext,nextWts.T)*(aL.input>0)


# INITIALIZE
layer1=layer(784,128)
layer2=layer(128,64)
layer3=layer(64,10)
activation1=ReLU()
activation2=ReLU()
activation3=softmax()
lossFunc=LCCE()
backPropOut=backPropL1()
backPropHid=backProp()
lr=0.01
batch_size=128

for i in range(500):
    ind=np.random.permutation(X_train.shape[0])
    X_shuffled=X_train[ind]
    y_shuffled=y_train[ind]
    for j in range(0,X_train.shape[0],batch_size):
        X=X_shuffled[j:j+batch_size]
        y=y_shuffled[j:j+batch_size]
        layer1.process(X)
        activation1.process(layer1.output)
        layer2.process(activation1.output)
        activation2.process(layer2.output)
        layer3.process(activation2.output)
        activation3.process(layer3.output)
        delta3=backPropOut.deltaCurr(y,activation3.output)
        dW3=backPropOut.deltaCalW(activation2.output,activation3.output,y)
        db3=backPropOut.deltaCalb(activation3.output,y)
        delta2=backPropHid.deltaCurr(activation2,layer3.weights,delta3)
        dW2=backPropHid.deltaCalW(activation1.output,activation2,layer3.weights,delta3)
        db2=backPropHid.deltaCalb(activation2,layer3.weights,delta3)
        delta1=backPropHid.deltaCurr(activation1,layer2.weights,delta2)
        dW1=backPropHid.deltaCalW(X,activation1,layer2.weights,delta2)
        db1=backPropHid.deltaCalb(activation1,layer2.weights,delta2)
        layer3.weights-=lr*dW3
        layer3.biases-=lr*db3
        layer2.weights-=lr*dW2
        layer2.biases-=lr*db2
        layer1.weights-=lr*dW1
        layer1.biases-=lr*db1
    layer1.process(X_train)
    activation1.process(layer1.output)
    layer2.process(activation1.output)
    activation2.process(layer2.output)
    layer3.process(activation2.output)
    activation3.process(layer3.output)
    lossVal=lossFunc.forward(activation3.output,y_train)
    predictions=np.argmax(activation3.output,axis=1)
    accuracy=np.mean(predictions==y_train)
    print(f"it {i+1}/500  loss: {lossVal:.4f}  accuracy: {accuracy*100:.2f}%")

    layer1.process(X_test)
    activation1.process(layer1.output)
    layer2.process(activation1.output)
    activation2.process(layer2.output)
    layer3.process(activation2.output)
    activation3.process(layer3.output)
    predictions=np.argmax(activation3.output,axis=1)
    test_accuracy=np.mean(predictions==y_test)
print(f"\nNeural Network ki accuracy: {test_accuracy*100:.2f}%")