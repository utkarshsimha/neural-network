import theano
import theano.tensor as T
import numpy as np
import gzip
import cPickle
import time

theano_rng = T.shared_randomstreams.RandomStreams( np.random.randint(2**30) )

class AutoEncoder( object ):

    def __init__( self, n_in, n_hid ):
        """
        || DESCRIPTION ||
        Implementation for a single stack of the Autoencoder for Pretraining 

        || PARAMS ||
        * n_in :
            type - int
            brief - Number of input neurons
        * n_hid :
            type - int
            brief - Number of hidden nuerons
        """
        ''' Initialize parameters of the autoencoder '''
        self.x = T.dmatrix('x')
        self.n_in = n_in
        self.n_hid = n_hid
        initial_W = np.asarray( np.random.uniform( 
            low=-np.sqrt( 6./(n_in + n_hid) ), high=np.sqrt( 6./(n_in + n_hid) ), size=(n_in,n_hid) ) )
        self.W = theano.shared(initial_W )
        self.W_prime = self.W.T
        self.B = theano.shared( np.zeros( ( n_hid, ) ) )
        self.B_prime = theano.shared( np.zeros( ( n_in, ) ) )
        self.params = [ self.W, self.B, self.B_prime ]


        ''' Computing the output of the autoencoder '''
        self.hiddenLayer = self.x.dot( self.W ) + self.B
        self.output = self.hiddenLayer.dot( self.W_prime ) + self.B_prime

        ''' Mean Squared Error cost function used '''
        L = ( self.x - self.output )**2
        self.cost =  T.mean( T.sum( L,axis=0 ) )

        ''' Stochastic gradient descent '''
        eta = T.dscalar('eta')
        updates = []
        for w in self.params:
            updates.append( ( w, w - eta * T.grad( self.cost, w ) ) )

        ''' Theano function to train the autoencoder '''
        self.train = theano.function( [self.x, eta], self.cost, updates=updates )

    def getHiddenLayerOutput( self, input ):
        """ Get the ouptut of the hidden layer to pass as input
        to the next stack """
        return theano.function( [self.x], self.hiddenLayer )( input )

    def getWeightsAndBias( self ):
        """ Return the weights and bias learnt by the autoencoder """
        return [ self.params[0], self.params[1] ]

if __name__ == '__main__':

    ae = AutoEncoder( 80, 10 )
    inp = np.random.random( ( 50, 80 ) )
    costs = []
    error = []
    for ep in range( 100 ):
    #while( True ):
        cost = ae.train( inp, eta=0.006 )
        print cost
        costs.append( cost )
        if( len(costs) > 2 and ( ( costs[-2] - costs[-1] < 0.00001 ) ) and ( ( costs[-2] - costs[-1] > 0 ) ) ):
            break
