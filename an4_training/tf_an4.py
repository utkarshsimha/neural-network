import numpy as np
import tensorflow as tf
import cPickle as pickle
import math

def loadDataset():
    ''' Load training, testing and validation dataset '''

    '''Training data''' 
    print "Loading Training dataset ..."
    inp, targ = pickle.load( open("train_data.p","rb") )
    print "Training dataset Loaded"

    '''Shuffle input'''
    print "Shuffling input ..."
    x = np.c_[inp.reshape(len(inp), -1), targ.reshape(len(targ), -1)]
    shuffle_inp = x[:, :inp.size//len(inp)].reshape(inp.shape)
    shuffle_targ = x[:, inp.size//len(inp):].reshape(targ.shape)
    np.random.shuffle( x )
    np.random.shuffle( x )
    inp = shuffle_inp
    targ = shuffle_targ
    print "Input shuffled"

    '''Testing data'''
    print "Loading Testing dataset ...."
    test_inp, test_targ = pickle.load( open("test_data.p","rb") )
    print "Testing dataset loaded"

    '''Validation data'''
    print "Loading Validation dataset ..."
    valid_inp = inp[ 200000: ]
    valid_targ = targ[ 200000: ]
    inp = inp[ :200000 ]
    targ = targ[ :200000 ]
    print "Validation dataset loaded"

    '''Organize into tuples'''
    train = ( inp, targ )
    test = ( test_inp, test_targ )
    valid = ( valid_inp, valid_targ )
    print 'Training set', train[0].shape, train[1].shape
    print 'Validation set', valid[0].shape, valid[1].shape
    print 'Test set', test[0].shape, test[1].shape
    return train, test, valid

class DeepNeuralNetwork:
    def __init__( self, n_in, n_out, test, valid, hiddenLayers, activation=tf.nn.sigmoid, batch_size=128, learning_rate=0.01 ):
        ''' Deep Neural Network implementation '''
        self.graph = tf.Graph()
        with self.graph.as_default():

            '''Training dataset, given in mini-batches '''
            self.tf_train = ( tf.placeholder( tf.float32, shape=(batch_size, n_in ) ), tf.placeholder( tf.float32, shape=(batch_size, n_out ) ) )

            ''' Validation dataset '''
            tf_valid = ( tf.cast( tf.constant( valid[0] ), tf.float32 ), tf.cast( tf.constant( valid[1] ), tf.float32 ) )

            '''Testing dataset'''
            tf_test = ( tf.cast( tf.constant( test[0] ), tf.float32 ), tf.cast( tf.constant( test[1] ), tf.float32 ) )

            self.weights = [] #Weights list
            self.bias = [] #Bias list
            self.l2_reg = 0. #L2 Regularization

            '''Inputs'''
            train_input = self.tf_train[0]
            valid_input = tf_valid[0]
            test_input = tf_test[0]

            layerIn = n_in #input to layer
            ''' Add hidden layers '''
            for layerOut, hdf in hiddenLayers:
                train_input = self._addLayer( train_input, layerIn, layerOut, activation=activation, dropout=hdf, l2_reg=True )
                valid_input = self._addLayer( valid_input, layerIn, layerOut, activation=activation, weights=self.weights[-1], bias=self.bias[-1] )
                test_input = self._addLayer( test_input, layerIn, layerOut, activation=activation, weights=self.weights[-1], bias=self.bias[-1] )
                ''' Input to next layer is output of current layer '''
                layerIn = layerOut

            '''Output layers '''
            train_logits = self._addLayer( train_input, layerIn, n_out )
            valid_logits = self._addLayer( valid_input, layerIn, n_out, weights=self.weights[-1], bias=self.bias[-1] )
            test_logits = self._addLayer( test_input, layerIn, n_out, weights=self.weights[-1], bias=self.bias[-1] )

            '''Cost function '''
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, self.tf_train[1])) + 0.0001 * self.l2_reg

            ''' Adagrad Optimizer '''
            self.optimizer = tf.train.AdagradOptimizer( learning_rate ).minimize( self.cost )

            ''' Prediction functions '''
            self.train_pred = tf.nn.softmax( train_logits )
            self.valid_pred = tf.nn.softmax( valid_logits )
            self.test_pred = tf.nn.softmax( test_logits )

    def _addLayer( self, input, n_in, n_out, activation=None, weights=None, bias=None, dropout=None, l2_reg=False ):
        if( weights is None ):
            ''' Xavier init '''
            init_range = math.sqrt(6.0 / (n_in + n_out))
            init_w = tf.random_uniform( [n_in,n_out], -init_range, init_range)
            weights = tf.cast( tf.Variable( init_w ), tf.float32 )
            self.weights.append( weights )
        if( bias is None ):
            bias = tf.cast( tf.Variable( tf.zeros( [ n_out ] ) ), tf.float32 )
            self.bias.append( bias )
        if( l2_reg ):
            ''' L2 regularization '''
            l2_reg = tf.nn.l2_loss( weights )
            self.l2_reg += l2_reg 

        ''' Affine transformation '''
        layer = tf.matmul( input, weights ) + bias 

        if( activation is not None ):
            layer = activation( layer )
        if( dropout is not None ):
            ''' Dropout + scaling '''
            layer = tf.nn.dropout( layer, 1-dropout ) * 1/( 1- dropout )
        return layer 

def accuracy( pred, labels ):
    return (100.0 * np.sum(np.argmax(pred, 1) == np.argmax(labels, 1)) / pred.shape[0])

if __name__ == '__main__':
    ''' Dataset '''
    train,valid,test = loadDataset()
    train_X = train[0]
    train_Y = train[1]

    ''' Params '''
    n_epochs = 600
    batch_size = 128
    learning_rate = 0.01
    num_layers = 3
    hiddenLayers = [ ( 1024, 0.5 ) ] * num_layers
    n_in = train_X.shape[1]
    n_out = train_Y.shape[1]
    n_batches = ( train_X.shape[0] / batch_size )
    activation = tf.nn.tanh

    ''' Model '''
    dnn = DeepNeuralNetwork( n_in, n_out, test, valid, hiddenLayers, activation=activation, batch_size=batch_size, learning_rate=learning_rate )

    with tf.Session( graph = dnn.graph ) as session:
        tf.initialize_all_variables().run()
        for ep in range( n_epochs ):
            avg_cost = 0.
            avg_acc = 0.
            valid_acc = []
            
            for mb_idx in range( n_batches ):
                ''' Mini-batching '''
                batch_X = train_X[ mb_idx*batch_size:(mb_idx + 1)*batch_size ]
                batch_Y = train_Y[ mb_idx*batch_size:(mb_idx + 1)*batch_size ]

                ''' Input to placeholders '''
                feed_dict = { dnn.tf_train[0]:batch_X, dnn.tf_train[1]:batch_Y }

                ''' Train step '''
                _, cost, train_pred = session.run( [ dnn.optimizer, dnn.cost, dnn.train_pred ], feed_dict=feed_dict )
                avg_cost += cost
                avg_acc += accuracy( train_pred, batch_Y )

            if( ep % 10 == 0 ):
                print "Cost at {} - {}".format( ep, avg_cost/n_batches )
                print "Training accuracy : {}".format( avg_acc/n_batches )
                val_acc = accuracy( dnn.valid_pred.eval(), valid[1] )
                print "Validation accuracy : {}".format( val_acc )
                valid_acc.append( val_acc )
                #TODO add patience for early stopping
                if( len(valid_acc) > 3 and int(valid_acc[-1]*100) - int(valid_acc[-3]*100) > -1 ):
                    break

        ''' Testing '''
        print "Test accuracy : {}".format( accuracy( dnn.test_pred.eval(), test[1] ) )
