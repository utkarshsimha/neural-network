import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T
from read_senid import *
from read_mfc import *
import autoencoder
import sys
''' Theano random number generator '''
theano_rng = T.shared_randomstreams.RandomStreams( np.random.randint(2**30) ) #Theano Random Number generator

class NeuralNetwork:

    def __init__( self, sizes, ae_wb_list=None, hdf=0.0, idf=0.0, reg_lambda=0.0001 ):
        """
            || DESCRIPTION ||
            Implementation of a Feed-Forward Neural Network
            with Stochastic Gradient Descent

            || PARAMS ||
            * sizes : 
                type - list
                brief - The architecture of the Neural Network. Sizes of each layer are passed as a list
            * ae_wb_list :
                type - list
                brief - The weights learnt by the autoencoder passed to initialize the Neural Network's weights
            * hdf :
                type - float
                brief - Hidden layer dropout factor
            * idf :
                type - float
                brief - Input layer dropout factor
            * reg_lambda :
                type - float
                brief - Lambda constant for L2 regularization

        """

        ''' Initialize the parameters of the Neural Network '''
        self.sizes = sizes
        self.wb_list = [] #List to store weights and bias
        self.x = T.dmatrix("x") #Input tensor
        self.output = T.dmatrix("output") #Output layer tensor
        self.output_test = T.dmatrix("output_test") #Output without dropout -- for testing
        self.labels = T.dmatrix("labels") #Actual output
        self.ae_wb_list = ae_wb_list #Weights and Bias obtained using autoencoder
        self.hdf = hdf 
        self.idf = idf

        ''' Create the Neural Network. Theano produces a computational graph
        for the tensors and its operations which is evaluated using a Theano function '''
        self.hiddenLayers = self.createNeuralNet() #Create neural network
        self.createNeuralNetNoDropout() #Create neural network graph without dropping out neurons -- for testing
        self.prediction = T.argmax(self.output_test, axis=1) #Predicted output of Neural Network

        ''' L2 regularization '''
        self.reg_lambda = reg_lambda 
        self.regularization = self.computeRegularization() #L2 regularization

        ''' Cost function uses a Cross-entropy function as the output unit is a Softmax layer '''
        self.cost = - T.mean( self.labels * T.log( self.output ) )

        ''' Theano functions for training and testing the Neural Network '''
        self.test = theano.function([self.x], self.prediction)
        self.train = self.computeTraining()
        

    def createNeuralNet( self ):
        """ Creates the Neural Network by initializing the model and producing the graph for
        the tensors with their operations """
        hiddenLayers = [] #To store the activations of each of the hidden layers
        self.initializeWeightsAndBias() 
        tilde_x = T.dmatrix("tilde_x") #Corrupted input
        for i in range( 0,len(self.wb_list)-1,2 ):
            if( i == 0 ): #Input layer -> Hidden Layer
                tilde_x = theano_rng.binomial( ndim=self.x.ndim,p=1-self.idf,n=1 ) * self.x #Input dropout
                hidden = T.nnet.sigmoid(tilde_x.dot(self.wb_list[i]) + self.wb_list[i+1]) 
                hiddenLayers.append( hidden )

            elif( i == len(self.wb_list)-2 ): #Hidden Layer -> Output layer
                output = T.nnet.softmax(hiddenLayers[-1].dot(self.wb_list[i]) + self.wb_list[i+1])
                self.output = output

            else: #Hidden layer -> Hidden Layer
                tilde_x = theano_rng.binomial( ndim=hiddenLayers[-1].ndim,p=1-self.hdf,n=1 ) * hiddenLayers[-1] 
                hidden = T.nnet.sigmoid(tilde_x.dot(self.wb_list[i]) + self.wb_list[i+1])
                hiddenLayers.append( hidden )

        return hiddenLayers

    def createNeuralNetNoDropout( self ):
        """ Create the Neural Network graph without dropping out any neurons. Used while testing """
        hiddenLayers = []
        ''' Scaling the input and hidden layer weights '''
        i_scale = 1-self.idf
        h_scale = 1-self.hdf
        for i in range( 0,len(self.wb_list)-1,2 ):
            if( i == 0 ): #Input layer -> Hidden layer
                hidden = T.nnet.sigmoid(self.x.dot(self.wb_list[i] * i_scale) + self.wb_list[i+1] * i_scale ) 
                hiddenLayers.append( hidden )
            elif( i == len(self.wb_list)-2 ): #Hidden Layer -> Output layer 
                output = T.nnet.softmax(hiddenLayers[-1].dot(self.wb_list[i] * h_scale) + self.wb_list[i+1] * h_scale)
                self.output_test = output
            else: #Hidden layer -> Hidden layer
                hidden = T.nnet.sigmoid(hiddenLayers[-1].dot(self.wb_list[i] * h_scale) + self.wb_list[i+1]* h_scale)
                hiddenLayers.append( hidden )
        return hiddenLayers

    def initializeWeightsAndBias( self ):
        """ Initialize the weights and bias terms. If weights from the autoencoder are present, use those weights
        to initialize the Neural Network. Otherwise initialize the weights using a uniform distribution """
        if( self.ae_wb_list is not None ):
            for w in self.ae_wb_list:
                self.wb_list.append( theano.shared( w.get_value() ) )
        else:
            for i in range( len(self.sizes)-1 ):
                W_shape = (self.sizes[i],self.sizes[i+1])
                B_shape = self.sizes[i+1]
                n_in = self.sizes[i]
                n_out = self.sizes[i+1]
                initial_W = np.asarray( np.random.uniform( 
                    low=-np.sqrt( 6./(n_in + n_out) ), high=np.sqrt( 6./(n_in + n_out) ), size=(n_in,n_out) ) )
                initial_W = 4*initial_W #Multiply by 4 for sigmoid function
                W = theano.shared( value=initial_W, name='W' )
                self.wb_list.append( W )
                B = theano.shared( value=np.zeros( B_shape ), name='B' )
                self.wb_list.append( B )

    def computeRegularization( self ):
        """ L2 regularization, to avoid overfitting 
        Compute the sum of square of the values for all weights and bias """
        self.reg = theano.shared(0.0)
        for w in self.wb_list:
            self.reg += T.sum( w**2 ) 
        return self.reg * self.reg_lambda

    def computeTraining( self ):
        """ Theano function to train the Neural Network 
        Backpropagating the error is done by using Theano's `grad` function """
        eta = T.dscalar("eta") #Learning rate
        momentum = T.dscalar("momentum") #Momentum parameter
        updates = [] #Weight and bias updates
        for w in self.wb_list:
            w_update = theano.shared( w.get_value()*0. )
            updates.append( ( w, w - eta * w_update ) )
            updates.append( ( w_update, momentum * w_update + ( 1. - momentum ) * T.grad( self.cost, w ) ) )
        train = theano.function([self.x, self.labels, eta, momentum], self.cost, updates=updates)
        return train


def loadDataset( mfcc_dir, senone_dir ):
    """ Load the dataset """
    data = []
    mfcc = readMelLogSpec( mfcc_dir )
    senones = read_segmentation( senone_dir )
    mfc_data = np.array([])
    sen_data = np.array([])
    i = 0
    """ Concatenate all the files together into a single dataset """
    for k in mfcc.keys():
        if k in senones.keys():
            if( i == 0 ):
                mfc_data = mfcc[k]
                sen_data = senones[k]
                i+=1
            else:
                mfc_data = np.concatenate((mfc_data,mfcc[k]))
                sen_data = np.concatenate((sen_data,senones[k]))

    return mfc_data,sen_data

def createOutputVec(labels, max_index):
    """ Given a vector, convert it to a vector where the values are represented with 1 in the argument of the value
    and 0 otherwise. Eg : 5 will be represented as [ 0 0 0 0 0 1 0 0 0 0 0 ] """
    vec = np.zeros((labels.shape[0], max_index + 1))
    for i in xrange(labels.shape[0]):
        vec[i, labels[i]] = 1
    return vec

def computeAccuracy(predicted, actual):
    """ Compute the accuracy, given the predicted value and the target value """
    tot = 0.0
    correct = 0.0
    for p, a in zip(predicted, actual):
        tot += 1
        b = a.tolist()
        if p == b.index(max(b)):
            correct += 1
    return correct / tot

def trainNeuralNet( nnet, train_input, train_label, batch_size, eta=0.03, epochs=1000, momentum=0.95):
    """ 
        || DESCRIPTION ||
        Train the Neural Network by dividing the input into minibatches

        || PARAMS ||
        * nnet
            type - NeuralNetwork object
            brief - Object of the Neural Network class to train upon
        * train_input
            type - numpy.ndarray
            brief - The input matrix for training
        * train_label
            type - numpy.ndarray
            brief - The target matrix for training
        * batch_size
            type - int
            brief - Batch size for creating mini-batches
        * eta
            type - float
            brief - Learning rate
        * epochs
            type - int
            brief - Number of iterations to train the Neural Network
        * momentum
            type - float
            brief - Momentum parameter

    """
    assert( momentum > 0 and momentum <= 1 )
    assert( train_input.shape[1] == nnet.sizes[0] )
    assert( train_label.shape[1] == nnet.sizes[-1] )
    n_batches = inp.shape[0] / batch_size
    costs = []
    u = eta #Base learning rate
    for i in range(1,epochs):
        cost_avg = []
        for idx in range( n_batches ): #Minibatches
            x = train_input[ idx*batch_size:(idx+1)*batch_size ]
            y = train_label[ idx*batch_size:(idx+1)*batch_size ]
            cost = nnet.train( x, y, eta, momentum )
            cost_avg.append( cost )
        costs.append( np.asarray( cost_avg ).mean() )
        print "Epoch", i, "\tCost : ",costs[-1]
        eta = u / ( 1 + 0.0001 * i ) #Learning rate decay
        if( len(costs) > 2 and ( costs[-2] - costs[-1] ) < 0.000001 ):
            break
    return costs

def testNeuralNet( nnet, test_set, label ):
    """ Test the Neural Network and return the predicted output from the
    Neural Network """
    pred = nnet.test(test_set)
    return createOutputVec(pred,101)


def autoEncode( sizes, input ):
    """ Pretraining using stacked autoencoders """
    assert( input.shape[1] == sizes[0] )
    ae_eta = 0.0000001
    n_epochs = 300
    batch_size = 1000 #Batch size
    n_batches = input.shape[0]/ batch_size #Minibatches for autoencoder training 
    ae_wb_list = [] #Weights and bias obtained from autoencoder
    nextStackInput = [] #Input for the next stack of the autoencoder (Output of previous hidden layer)
    for i in range( len(sizes)-1 ):
        inpLayer = sizes[i]
        hidLayer = sizes[i+1]
        ae = autoencoder.AutoEncoder( inpLayer, hidLayer )
        print "Running stack",i+1,"of autoencoding"
        if i == 0:
            for ep in range( n_epochs ):
                cost_avg = []
                for idx in range( n_batches ):
                    x = input[ idx*batch_size:(idx+1)*batch_size ]
                    cost_avg.append(ae.train( x, eta=ae_eta ))
                print "Epoch",ep,"Cost :",(sum(cost_avg)/len(cost_avg))
            ''' Get output from the hidden layer to pass as input for the next stack '''
            nextStackInput = ae.getHiddenLayerOutput( input )
        else:
            input = nextStackInput
            for ep in range( n_epochs ):
                cost_avg = []
                for idx in range( n_batches ):
                    x = input[ idx*batch_size:(idx+1)*batch_size ]
                    cost_avg.append(ae.train( x, eta=ae_eta ))
                print "Epoch",ep,"Cost :",(sum(cost_avg)/len(cost_avg))
            if( i != len(sizes) -2 ):
                nextLayerInput = ae.getHiddenLayerOutput( input )
        """ Get the weights and bias learnt by the autoencoder """
        for wb in ae.getWeightsAndBias():
            ae_wb_list.append( wb )
    return ae_wb_list


if __name__ == '__main__':
    if( len( sys.argv ) < 5):
	sys.exit('Usage %s  EPOCHS ETA BATCHSIZE HDF IDF NETWORK MOMENTUM REG_LAMBDA' % sys.argv[0])
    EPOCHS = int(sys.argv[1])
    ETA = float(sys.argv[2])
    BATCHSIZE = int(sys.argv[3])
    HDF = float(sys.argv[4])
    IDF = float(sys.argv[5])
    NETWORK = (sys.argv[6]).split(',')
    NETWORK = [ int(i) for i in NETWORK ]
    MOMENTUM = float(sys.argv[7])
    REG_LAMBDA =float(sys.argv[8])

    print "Loading Training dataset ..."
    #train_data = loadDataSet( 'an4_clstk', 'st_seg_train_ci' )
    inp = cPickle.load(open('med_train.p','rb'))
    targ = cPickle.load(open('med_train_sen.p','rb'))
    print "Training dataset Loaded"
    
    ''' Shuffle the input '''
    x = np.c_[inp.reshape(len(inp), -1), targ.reshape(len(targ), -1)]
    shuffle_inp = x[:, :inp.size//len(inp)].reshape(inp.shape)
    shuffle_targ = x[:, inp.size//len(inp):].reshape(targ.shape)
    np.random.shuffle( x )
    np.random.shuffle( x )
    np.random.shuffle( x )

    ae_wb_list = None
    ''' Pretraining using Stacked Autoencoders '''
    ae_wb_list = autoEncode( NETWORK, inp )

    batch_size = BATCHSIZE
    #Create Neural Network object
    nnet = NeuralNetwork( sizes=NETWORK, ae_wb_list=ae_wb_list, hdf=HDF, idf=IDF, reg_lambda=REG_LAMBDA )

    #Train Neural Network
    cost = trainNeuralNet( nnet=nnet, train_input=inp, train_label=targ, batch_size=batch_size, epochs=EPOCHS, eta=ETA, momentum=MOMENTUM)

    #Computing training dataset accuracy
    pred = nnet.test( inp )
    print "Training accuracy : ",computeAccuracy( pred, targ )

    print "Loading Testing dataset ...."
    #train_data = loadDataSet( 'an4test_clstk', 'st_seg_test_ci' )
    test_inp = cPickle.load( open('large_test.p','rb') )
    test_targ = cPickle.load( open('large_test_sen.p','rb') )
    print "Testing dataset loaded"

    #Computing testing dataset accuracy
    pred = nnet.test( test_inp )
    print "Testing accuracy : ",computeAccuracy( pred, test_targ )
