import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import pool as downsample
import numpy as np
from utility import *


class FullyConnectedLayer(object):
    def __init__(
        self,
        rng,
        input=None, 
        ninput=784,
        noutput=10,
        activation=None,
        W=None,
        b=None
    ):

        self.input = input

        if W is None:
            Wvalues = np.asarray(
                rng.uniform(low=-np.sqrt(6.0/(ninput+noutput)),
                            high=np.sqrt(6.0/(ninput+noutput)),
                            size=(ninput, noutput)),
                dtype=theano.config.floatX)
            if activation == sigmoid:
                Wvalues *= 4.0
            W = theano.shared(value=Wvalues, name='W', borrow=True)

        if b is None:
            bvalues = np.zeros((noutput,), dtype=theano.config.floatX)
            b = theano.shared(value=bvalues, name='b', borrow=True)

        self.W = W
        self.b = b

        linearoutput = T.dot(input, self.W) + self.b

        if activation is None:
            self.output = linearoutput
        else:
            self.output = activation(linearoutput)

        self.params = [self.W, self.b]

class EmbedIDLayer(object):

    def __init__(
        self,
        rng,
        input=None,
        ninput=None,
        noutput=None,
        W=None,
    ):
        if input is None:
            input = T.imatrix('x')

        if W is None:
            Wvalues = np.asarray(
                rng.uniform(low=-np.sqrt(6.0/(ninput+noutput)), high=np.sqrt(6.0/(ninput+noutput)), size=(ninput, noutput)),    dtype=theano.config.floatX)

            # W_values[0,0] = 0
            Wtmp = theano.shared(value=Wvalues, name='W', borrow=True)
        else:
            Wvalues = W.astype(theano.config.floatX)
            Wtmp = theano.shared(value=Wvalues, name='W', borrow=True)

        self.W = Wtmp
        self.output = self.W[input]
        self.params = [self.W]


        
class MaxPoolingLayer(object):
    def __init__(self, input, poolsize=(2,2)):
        pooledout = downsample.pool_2d(
            input=input,
            ws=poolsize,
            ignore_border=True
        )
        self.output = pooledout        


class ConvolutionalLayer(object):
    def __init__(
        self,
        rng,
        input,
        filter_shape=None,
        image_shape=None,
        activation=relu
    ):
        self.input = input
        self.rng = rng

        fanin = np.prod(filter_shape[1:]) # 1*2*3
        fanout = filter_shape[0] * np.prod(filter_shape[2:]) # 0*2*3
        
        Wbound = np.sqrt(6.0 / (fanin + fanout))
        self.W = theano.shared(
            np.asarray(
                self.rng.uniform(
                    low = -Wbound,
                    high = Wbound,
                    size = filter_shape
                ),
                dtype = theano.config.floatX
            ),
            borrow = True
        )

        bvalues = np.zeros((filter_shape[0],),
                            dtype=theano.config.floatX)
        self.b = theano.shared(bvalues, borrow=True)

        convout = conv.conv2d(
            input=self.input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        self.output = activation(convout + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]









