import theano
import theano.tensor as T
import numpy as np
import os
import matplotlib.pyplot as plt
class RecurrentNeuralNetwork:
	def __init__( self, sizes, activation ):
		self.sizes = sizes
		self.activation = activation
		self.x = T.dmatrix('x')
		self.y = T.dmatrix('y')
		self.eta = T.dscalar('eta')
		self.output = T.dmatrix('output')
		self.createRNN()
		self.cost = T.nnet.binary_crossentropy(self.output, self.y).mean()
		self.train = self.computeTraining() 

	def createRNN( self ):
		self.initNetworkParams()
		self.hiddenLayers = []
		for it in range( len(self.sizes)-1 ):
			W = self.interLayerWeights[ it ]
			W_prime = self.intraLayerWeights[ it ]
			B = self.bias[ it ]
			H0 = self.initH[ it ]
			if( it == 0 ): #input layer
				h, _ = theano.scan( self.recurrence, sequences=self.x, outputs_info=[H0], non_sequences=[ W, W_prime, B ] )
				self.hiddenLayers.append( h )
			elif( it == len(self.sizes)-2 ): #output layer
				self.output = T.nnet.softmax( self.hiddenLayers[-1].dot( W ) + B )	
			else: #hidden layer
				h, _ = theano.scan( self.recurrence, sequences=self.hiddenLayers[-1], outputs_info=[H0], non_sequences=[ W, W_prime, B ] )
				self.hiddenLayers.append( h )

	def recurrence( self, x_t, h_tm1, W, W_prime, B ):
		return self.activation( x_t.dot( W ) + h_tm1.dot( W_prime )  + B )
				
	def initNetworkParams( self ):
		self.interLayerWeights = []
		self.intraLayerWeights = []
		self.bias = []
		self.initH = []
		for it in range( len(self.sizes)-1 ):
			n_in = self.sizes[it]
			n_out = self.sizes[it+1]
			init_W = np.random.random( ( n_in, n_out ) ) - 0.5
			init_W_prime= np.random.random( ( n_out, n_out ) ) - 0.5
			init_B = np.zeros( ( n_out, ), dtype=theano.config.floatX )
			h0 = theano.shared( np.zeros((n_out,), dtype=theano.config.floatX), 'h0' )
			self.interLayerWeights.append( theano.shared( init_W, 'W' ) )
			self.intraLayerWeights.append( theano.shared( init_W_prime, 'W_prime' ) )
			self.bias.append( theano.shared( init_B, 'B' ) )
			self.initH.append( h0 )

	def computeTraining( self ):
		updates = []
		params = self.interLayerWeights + self.intraLayerWeights[:len(self.intraLayerWeights)-1] + self.bias + self.initH[:len(self.initH)-1] 
		for w in params:
			updates.append( ( w, w - self.eta*T.grad( self.cost, w ) ) )
		train = theano.function( [ self.x, self.y, self.eta ], self.cost, updates=updates )
		return train



if __name__ == '__main__':
	rnn = RecurrentNeuralNetwork( [ 2, 3, 4, 5 ], T.nnet.sigmoid )	
	inp = [ ]
	inp = ( np.random.random( ( 10000, 2 ) ), np.random.random( ( 10000, 5 ) ) )
	costs = []
	for i in range( 100 ):
		costs.append( rnn.train( inp[0], inp[1], 0.3 ) )
		print costs[-1]
	plt.plot( range(len(costs)), costs )
	plt.show()
