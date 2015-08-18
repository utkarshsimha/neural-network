import theano
import theano.tensor as T
import numpy as np
import gzip
import cPickle
import time

theano_rng = T.shared_randomstreams.RandomStreams( np.random.randint(2**30) )

class AutoEncoder( object ):

    def __init__( self, n_in, n_out ):

        self.x = T.dmatrix('x')
        self.n_in = n_in
        self.n_out = n_out

        initial_W = np.asarray( np.random.uniform( 
            low=-np.sqrt( 6./(n_in + n_out) ), high=np.sqrt( 6./(n_in + n_out) ), size=(n_in,n_out) ) )
        self.W = theano.shared(initial_W )
        self.W_prime = theano.shared( self.W.get_value().transpose() )

        self.B = theano.shared( np.zeros( ( n_out, ) ) )
        self.B_prime = theano.shared( np.zeros( ( n_in, ) ) )

        self.params = [ self.W, self.B, self.W_prime, self.B_prime ]

        self.hiddenLayer = self.x.dot( self.W ) + self.B
        self.output = self.hiddenLayer.dot( self.W_prime +self.B_prime )

        L = ( self.x - self.output )**2
        self.getError = theano.function( [self.x], T.sum( L,axis=0 ) )
        self.cost =  T.mean( T.sum( L,axis=0 ) )

        gparams = []
        for w in self.params:
            gparams.append( T.grad( self.cost, w ) )

        self.print_gparams = theano.function( [ self.x ], gparams )
        eta = T.dscalar('eta')
        updates = []
        for w,dw in zip( self.params, gparams ):
            updates.append( ( w, w - eta * dw ) )

        self.train = theano.function( [self.x, eta], self.cost, updates=updates )


    def getHiddenLayerOutput( self, input ):
        return theano.function( [self.x], self.hiddenLayer )( input )

    def getWeightsAndBias( self ):
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
        error.append( ae.getError( inp ) )
