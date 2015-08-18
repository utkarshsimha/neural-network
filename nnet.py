'''
To implement a feed foward Neural Network with Stochastic Gradient Descent
for recognizing handwritten digits from the MNIST database

@author : Utkarsh Simha ( utkarshsimha@gmail.com )
'''
import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T
import autoencoder 

theano_rng = T.shared_randomstreams.RandomStreams( np.random.randint(2**30) ) #Theano Random Number generator

class NeuralNetwork:

    def __init__( self, sizes, ae_wb_list=None, do_rate=0.0 ):
        """

        || DESCRIPTION || 
        Implentation of a feed forward Neural Network.
        The neurons use a sigmoid activation function

        || PARAMETERS ||
        :type sizes: list
        :param sizes: The size of the layers of the Neural Network is passed as a list.

        :type ae_wb_list: list
        :param ae_wb_list: Weights obtained from the autoencoder can be used to initialize the Neural Network.
                           Even pickled weights can be passed.

        :type do_rate: float
        :param do_rate: The dropout rate for randomly dropping out neurons from the layers of the nueral network

        """

        '''Initializing parameters of the Neural Network'''
        self.sizes = sizes
        self.wb_list = [] #List to store weights and bias
        self.x = T.dmatrix("x") #Input tensor
        self.output = T.dmatrix("output") #Output layer tensor
        self.labels = T.dmatrix("labels") #Expected Output tensor (Target)
        self.ae_wb_list = ae_wb_list #Weights and Bias obtained from autoencoder
        self.do_rate = do_rate #Dropout rate

        ''' Creating the neural network using theano tensors. Theano uses lazy evaluation.
            The computational graphs for the tensors are intialized at this point. '''
        self.hiddenLayers = self.createNeuralNet() #Create neural network
        self.out_test = T.dmatrix("out_test") #Output without dropout -- for testing
        self.createNNetNoDropout() 

        self.feedForward = theano.function( [ self.x ], self.output ) #Feed Forward
        ''' The output of the Neural Network is the argmax of the ouput neurons '''
        self.prediction = T.argmax(self.out_test, axis=1) #Output of the Neural Network

        ''' L2 Norm Regularization '''
        self.reg_lambda = 0.0001 #Lambda constant for regularization
        self.regularization = self.computeRegularization() #L2 regularization

        ''' Cost function uses the binary crossentropy divergence. This means that larger the difference between the output and target
            , higher the cost reduction and vice-versa. '''
        self.cost = T.nnet.binary_crossentropy(self.output, self.labels).mean() + self.regularization #Cost function

        ''' Theano functions for training and computing the prediction of the Neural Network '''
        self.computePrediction = theano.function([self.x], self.prediction)
        self.train = self.computeTraining()
        
    def createNeuralNet( self ):
        ''' Function to create the Neural Network using Theano's Tensor Variables 
        Activation function of the Neural Network is the sigmoid function
        
        '''
        hiddenLayers = [] 
        self.initializeWeightsAndBias()
        self.tilde_x = T.dmatrix('tilde_x') #Corrupted input
        for i in range( 0, len(self.wb_list)-1, 2 ):
            if( i == 0 ): #First layer
                x = self.x 
                self.tilde_x = theano_rng.binomial( ndim=x.ndim,p=1-self.do_rate,n=1 ) * x #Randomly set a few neurons to 0
                hidden = T.nnet.sigmoid(self.tilde_x.dot(self.wb_list[i]) + self.wb_list[i+1])
                #hidden = T.nnet.sigmoid(x.dot(self.wb_list[i]) + self.wb_list[i+1]) #Without dropout
                hiddenLayers.append( hidden )
            elif( i == len(self.wb_list)-2 ): #Output layer
                output = T.nnet.softmax(hiddenLayers[-1].dot(self.wb_list[i]) + self.wb_list[i+1]) #Softmax function 
                self.output = output
            else: #Hidden layers
                self.tilde_x = theano_rng.binomial( ndim=hiddenLayers[-1].ndim,p=1-do_rate,n=1 ) * hiddenLayers[-1]
                hidden = T.nnet.sigmoid(self.tilde_x.dot(self.wb_list[i]) + self.wb_list[i+1])
                #hidden = T.nnet.sigmoid(hiddenLayers[-1].dot(self.wb_list[i]) + self.wb_list[i+1]) #Without Dropout
                hiddenLayers.append( hidden )
        return hiddenLayers

    def createNNetNoDropout( self ):
        ''' Create Neural Network without dropping out neurons. Used for testing '''

        #Scaling of weights
        if( self.do_rate == 0.0 ):
            scale = 1.0
        else:
            scale = 1/self.do_rate
        hiddenLayers = []
        for i in range( 0,len(self.wb_list)-1,2 ):
            if( i == 0 ): #input layer
                x = self.x 
                hidden = T.nnet.sigmoid(x.dot(self.wb_list[i] * scale) + self.wb_list[i+1] * scale)
                hiddenLayers.append( hidden )
            elif( i == len(self.wb_list)-2 ): #output layer
                output = T.nnet.softmax(hiddenLayers[-1].dot(self.wb_list[i] * scale) + self.wb_list[i+1] * scale)
                self.out_test = output
            else: #hidden layers
                hidden = T.nnet.sigmoid(hiddenLayers[-1].dot(self.wb_list[i] * scale) + self.wb_list[i+1] * scale)
                hiddenLayers.append( hidden )
     

    def initializeWeightsAndBias( self ):
        ''' Initializes the weights and bias terms. If weights from the autoencoder are present, initialize using those weights.
        Else initialize randomly using a uniform distribution '''

        if( self.ae_wb_list is not None ): #If pretraining is done
            for w in self.ae_wb_list:
                self.wb_list.append( theano.shared( w.get_value() ) ) #Weights from autoencoder
        else: #Random initialization
            for i in range( len(self.sizes)-1 ):
                W_shape = (self.sizes[i],self.sizes[i+1]) #Shape : nth layers neuron's X n+1th layer's neurons
                B_shape = self.sizes[i+1]
                initial_W = np.asarray( np.random.uniform( 
                    low=-np.sqrt( 6./(n_in + n_out) ), high=np.sqrt( 6./(n_in + n_out) ), size=(n_in,n_out) ) )
                initial_W = 4*initial_W #Multiply by 4 for sigmoid function
                W = theano.shared( initial_W  )
                self.wb_list.append( W )
                B = theano.shared( np.zeros( B_shape ) )
                self.wb_list.append( B )

    def computeRegularization( self ):
        ''' L2 regularization, to avoid overfitting '''
        self.reg = theano.shared(0.0)
        for w in self.wb_list:
            self.reg += (w*w).sum() #Sum of square of weights
        return self.reg * self.reg_lambda

    def computeTraining( self ):
        ''' Compute the theano function for training '''
        alpha = T.dscalar("alpha") #Learning rate
        updates = [(w, w - alpha * theano.grad(self.cost, w)) for w in self.wb_list] #Backpropogation  (Update for weights)
        train = theano.function([self.x, self.labels, alpha], self.cost, updates=updates)
        return train



def createOutputVec( labels, num_classes ):
    ''' Given a vector, represent it as a bit matrix where the element indexed by the value
    is set to 1 and 0 otherwise. Eg : 5 will be represented as [ 0 0 0 0 0 1 0 0 0 0 0 ] 
    num_classes corresponds to the number of classes in your output ( Eg. MNIST - 10 )
    '''

    vec = np.zeros((labels.shape[0], num_classes))
    for i in xrange(labels.shape[0]):
        vec[i, labels[i]] = 1
    return vec

def computeAccuracy(predicted, actual):
    ''' Compute the accuracy given the prediction and the target '''
    tot = 0.0
    correct = 0.0
    for p, a in zip(predicted, actual):
        tot += 1
        if p == a:
            correct += 1
    return correct / tot

def loadDataset():
    ''' Load the dataset '''
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)

    print "Shapes of input :"
    print "Training:   ",train_set[0].shape, train_set[1].shape
    print "Test:       ",test_set[0].shape, test_set[1].shape
    return train_set, test_set

def autoEncode( sizes, input, autoencoder_eta=0.000001 ):
    ''' Pretraining using stacked autoencoders '''
    ae_wb_list = []
    for i in range( len(sizes)-1 ):
        inpLayer = sizes[i]
        hidLayer = sizes[i+1]
        print "Running stack",i+1,"of autoencoding"
        costs = []
        cost = 0
        ae = autoencoder.AutoEncoder( inpLayer, hidLayer ) #Create autoencoder object
        if i == 0:
            for ep in range(400): #400 Epochs
                cost = ae.train( input, eta=autoencoder_eta ) #Train the autoencoder
                print "Epoch",ep,"Cost :",cost
                costs.append( cost )
            #Pass the output of the hidden layer of current stack as input to the next stack
            input  = ae.getHiddenLayerOutput(input)
        else:
            for ep in range( 400 ):
                cost = ae.train( input, eta=autoencoder_eta  )
                print "Epoch",ep,"Cost :",cost
                costs.append( cost )
            if( i != len(sizes) -2 ):
                input = ae.getHiddenLayerOutput( input )

        #Get weights and bias learnt by the autoencoder
        for wb in ae.getWeightsAndBias():
            ae_wb_list.append( wb )
    return ae_wb_list


        

if __name__ == '__main__':
    eta = 0.3 #Learning rate
    batch_size = 50 #Batch size
    nnet_architecture = [ 784, 50, 10 ] #Architecture of neural network
    max_epochs = 3000 #Maximum epochs. If convergence condition is satisfied, training will stop before this is reached
    do_rate = 0.15 #Dropout rate
    autoencoder_eta = 0.000001 #Learning rate for autoencoder

    print "Learning rate : " ,eta ,"Batch size : "\
    ,batch_size ,"\nNNet architecture : " ,nnet_architecture, "Dropout rate : ",\
    do_rate, "Autoendoder learning rate : ", autoencoder_eta

    train_set, test_set = loadDataset()
    labeled = createOutputVec(train_set[1], 10) #Target output (Expected output)
    train_set_x = train_set[0] #Training data
    
    n_batches = train_set_x.shape[0] / batch_size #Number of batches
    
    ae_wb_list = autoEncode( nnet_architecture, train_set[0] ) #Pretraining

    nnet = NeuralNetwork( nnet_architecture, ae_wb_list=ae_wb_list, do_rate=do_rate) #Create Neural Network
    costs = []
    i = 0
    while( i < max_epochs ):
        i += 1
        cost_avg = []
        for mb_idx in range( n_batches ): #Minibatches
            x = train_set_x[mb_idx*batch_size:(mb_idx+1)*batch_size]
            y = labeled[mb_idx*batch_size:(mb_idx+1)*batch_size]
            cost_avg.append( nnet.train(x,y,eta) )
        cost = np.asarray( cost_avg ).mean()
        costs.append(cost)
        print "Epoch", i, "\tCost : ",cost
        if ( len(costs) > 2 ) and ( abs( costs[-2] - costs[-1] ) < 0.00001 ): #Check for convergence
            if eta < 0.01 and ( abs( costs[-2] - costs[-1] ) < 0.000001 ) : #Stopping condition
                break
            else:
                eta = eta / 1.1 #Learning rate decay

    pred = nnet.computePrediction(test_set[0]) #Testing
    print "Prediction accuracy : ",computeAccuracy(pred, test_set[1]) #Print accuracy
