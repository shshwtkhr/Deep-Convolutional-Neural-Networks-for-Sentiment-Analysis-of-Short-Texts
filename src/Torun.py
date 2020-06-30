from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T

from sklearn.cross_validation import train_test_split
from theano.tensor.nnet import conv
from theano.tensor.signal import pool as  downsample

from Architecture import ConvolutionalLayer
from Architecture import EmbedIDLayer
from Architecture import FullyConnectedLayer
from Architecture import MaxPoolingLayer
from optimizers import *
from utility import *


class CharSCNN(object):
    def __init__(
        self,
        rng,
        batchsize=100,
        activation=relu
    ):
        
        import loader
        (numsent, charcnt, wordcnt, maxwordlen, maxsenlen,\
        kchr, kwrd, xchr, xwrd, y) = loader.read("tweets_clean.txt")

        dimword = 30
        dimchar = 5
        clword = 300
        clchar = 50
        kword = kwrd
        kchar = kchr

        datatrainword,\
        datatestword,\
        datatrainchar,\
        datatestchar,\
        targettrain,\
        targettest\
        = train_test_split(xwrd, xchr, y, random_state=1234, test_size=0.1)

        xtrainword = theano.shared(np.asarray(datatrainword, dtype='int16'), borrow=True)
        xtrainchar = theano.shared(np.asarray(datatrainchar, dtype='int16'), borrow=True)
        ytrain = theano.shared(np.asarray(targettrain, dtype='int8'), borrow=True)
        xtestword = theano.shared(np.asarray(datatestword, dtype='int16'), borrow=True)
        xtestchar = theano.shared(np.asarray(datatestchar, dtype='int16'), borrow=True)
        ytest = theano.shared(np.asarray(targettest, dtype='int8'), borrow=True)


        self.ntrainbatches = xtrainword.get_value(borrow=True).shape[0] / batchsize
        self.ntestbatches = xtestword.get_value(borrow=True).shape[0] / batchsize


        
        index = T.iscalar()
        xwrd = T.wmatrix('xwrd')
        xchr = T.wtensor3('xchr')
        y = T.bvector('y')
        train = T.iscalar('train')


        layercharembedinput = xchr

        layercharembed = EmbedIDLayer(
            rng,
            layercharembedinput,
            ninput=charcnt,
            noutput=dimchar
        )

        layer1input = layercharembed.output.reshape(
            (batchsize*maxsenlen, 1, maxwordlen, dimchar)
        )

        layer1 = ConvolutionalLayer(
            rng,
            layer1input,
            filter_shape=(clchar, 1, kchar, dimchar),
            image_shape=(batchsize*maxsenlen, 1, maxwordlen, dimchar)
        )

        layer2 = MaxPoolingLayer(
            layer1.output,
            poolsize=(maxwordlen-kchar+1, 1)
        )

        layerwordembedinput = xwrd

        layerwordembed = EmbedIDLayer(
            rng,
            layerwordembedinput,
            ninput=wordcnt,
            noutput=dimword
        )

        layer3wordinput = layerwordembed.output.reshape((batchsize, 1, maxsenlen, dimword))
        layer3charinput = layer2.output.reshape((batchsize, 1, maxsenlen, clchar))


        layer3input = T.concatenate(
            [layer3wordinput,
             layer3charinput],
            axis=3
        )


        layer3 = ConvolutionalLayer(
            rng,
            layer3input,
            filter_shape=(clword, 1, kword, dimword + clchar),
            image_shape=(batchsize, 1, maxsenlen, dimword + clchar),
            activation=activation
        )

        layer4 = MaxPoolingLayer(
            layer3.output,
            poolsize=(maxsenlen-kword+1, 1)
        )

        layer5input = layer4.output.reshape((batchsize, clword))

        layer5 = FullyConnectedLayer(
            rng,
            dropout(rng, layer5input, train),
            ninput=clword,
            noutput=50,
            activation=activation
        )

        layer6input = layer5.output

        layer6 = FullyConnectedLayer(
            rng,
            dropout(rng, layer6input, train, p=0.1),
            ninput=50,
            noutput=2,
            activation=None
        )

        result = Result(layer6.output, y)
        loss = result.negativeloglikelihood()
        accuracy = result.accuracy()
        params = layer6.params\
                +layer5.params\
                +layer3.params\
                +layerwordembed.params\
                +layer1.params\
                +layercharembed.params
        updates = RMSprop(learningrate=0.001, params=params).updates(loss)

        self.trainmodel = theano.function(
            inputs=[index],
            outputs=[loss, accuracy],
            updates=updates,
            givens={
                xwrd: xtrainword[index*batchsize: (index+1)*batchsize],
                xchr: xtrainchar[index*batchsize: (index+1)*batchsize],
                y: ytrain[index*batchsize: (index+1)*batchsize],
                train: np.cast['int32'](1)
            }
        )

        self.testmodel = theano.function(
            inputs=[index],
            outputs=[loss, accuracy],
            givens={
                xwrd: xtestword[index*batchsize: (index+1)*batchsize],
                xchr: xtestchar[index*batchsize: (index+1)*batchsize],
                y: ytest[index*batchsize: (index+1)*batchsize],
                train: np.cast['int32'](0)
            }
        )


    def trainandtest(self, nepoch=4):
        epoch = 0
        accuracies = []
        while epoch < nepoch:
            epoch += 1
            sumloss = 0
            sumaccuracy = 0
            for batchindex in xrange(self.ntrainbatches):
                batchloss, batchaccuracy = self.trainmodel(batchindex)
                sumloss = 0
                sumaccuracy = 0
                for batchindex in xrange(self.ntestbatches):
                    batchloss, batchaccuracy = self.testmodel(batchindex)
                    sumloss += batchloss
                    sumaccuracy += batchaccuracy
                loss = sumloss / self.ntestbatches
                accuracy = sumaccuracy / self.ntestbatches
                accuracies.append(accuracy)

                print('epoch: {}, test mean loss={}, test accuracy={}'.format(epoch, loss, accuracy))
                print('')
        return accuracies


if __name__ == '__main__':
    random_state = 1234
    rng = np.random.RandomState(random_state)
    charscnn = CharSCNN(rng, batchsize=10, activation=relu)
    charscnn.trainandtest(nepoch=3)
    






















