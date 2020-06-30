from __future__ import print_function

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import pool as downsample
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

def relu(x):
    return T.maximum(0, x)
def sigmoid(x):
    return T.nnet.sigmoid(x)
def tanh(x):
    return T.tanh(x)

def dropout(rng, x, train, p=0.5):
    maskedx = None
    if p > 0.0 and p < 1.0:
        seed = rng.randint(2 ** 30)
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
        mask = srng.binomial(
            n=1,
            p=1.0-p,
            size=x.shape,
            dtype=theano.config.floatX
        )
        maskedx = x * mask
    else:
        maskedx = x
    return T.switch(T.neq(train, 0), maskedx, x*(1.0-p))

def getcorruptedinput(rng, x, train, corruptionlevel=0.3):
    masked_x = None
    if corruptionlevel > 0.0 and corruptionlevel < 1.0:
        seed = rng.randint(2 ** 30)
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
        mask = srng.binomial(
            n=1,
            p=1.0-corruptionlevel,
            size=x.shape,
            dtype=theano.config.floatX
        )
        maskedx = x * mask
    return T.switch(T.neq(train, 0), maskedx, x)

def maxpool2d(x, poolsize=(2,2)):
    pooledout = downsample.pool2d(
        input = x,
        ws = poolsize,
        ignoreborder = True
    )
    return pooled_out

def embedid(sentences=None, nvocab=None, kwrd=None):
    if sentences is None or nvocab is None or kwrd is None:
        return NotImplementedError()

    tmp = sentences.getvalue(borrow=True)
    maxsentlen = len(tmp[0])
    xwrd = []
    for sentence in tmp:
        wordmat = np.array([[0]*nvocab]*(maxsentlen+kwrd-1), dtype='int8')

        i = 0
        for word in sentence:
            wordmat[(kwrd/2)+i][word] = 1
            i += 1

        xwrd.append(wordmat)
    return theano.shared(xwrd, borrow=False)



class Result(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def negativeloglikelihood(self):
        self.probofygivenx = T.nnet.softmax(self.x)
        return -T.mean(T.log(self.probofygivenx)[T.arange(self.y.shape[0]), self.y])

    def crossentropy(self):
        self.probofygivenx = T.nnet.softmax(self.x)
        return T.mean(T.nnet.categoricalcrossentropy(self.probofygivenx, self.y))

    def meansquarederror(self):
        return T.mean((self.x - self.y) ** 2)

    def errors(self):
        if self.y.ndim != self.ypred.ndim:
            raise TypeError('y should have the same shape as self.ypred',
                            ('y', self.y.type, 'ypred', self.ypred.type))

        if self.y.dtype.startswith('int'):
            self.probofygivenx = T.nnet.softmax(self.x)
            self.ypred = T.argmax(self.probofygivenx, axis=1)
            return T.mean(T.neq(self.ypred, self.y))
        else:
            return NotImplementedError()

    def accuracy(self):
        if self.y.dtype.startswith('int'):
            self.probofygivenx = T.nnet.softmax(self.x)
            self.ypred = T.argmax(self.probofygivenx, axis=1)
            return T.mean(T.eq(self.ypred, self.y))
        else:
            return NotImplementedError()

def loaddata(randomstate=0):
    print('fetch MNIST dataset')
    mnist = fetch_mldata('MNIST original')
    mnist.data   = mnist.data.astype(np.float32)
    mnist.data  /= 255
    mnist.target = mnist.target.astype(np.int32)

    datatrain,\
    datatest,\
    targettrain,\
    targettest\
    = traintestsplit(mnist.data, mnist.target, randomstate=randomstate)

    def shareddata(x,y):
        sharedx = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
        sharedy = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=True)

        return sharedx, T.cast(sharedy, 'int32')

    datatrain, targettrain = shareddata(datatrain, targettrain)
    datatest, targettest = shareddata(datatest, targettest)

    return ([datatrain, datatest], [targettrain, targettest])

def loadlivedoornewscorpus(randomstate=0, testsize=0.1):
    import six.moves.cPickle as pickle
    data_, target_ = pickle.load(open('dataset', 'rb'))

    datatrain,\
    datatest,\
    targettrain,\
    targettest\
    = traintestsplit(data, target, randomstate=randomstate, testsize=testsize)

    def shareddata(x,y):
        sharedx = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
        sharedy = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=True)

        return sharedx, T.cast(sharedy, 'int32')

    datatrain, targettrain = shareddata(datatrain, targettrain)
    datatest, targettest = shareddata(datatest, targettest)

    return ([datatrain, datatest], [targettrain, targettest])


def buildsharedzeros(shape, name):
    return theano.shared(
        value=np.zeros(shape, dtype=theano.config.floatX), 
        name=name, 
        borrow=True
    )

















