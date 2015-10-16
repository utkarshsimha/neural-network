import os
import re
import numpy as np
import time
import cPickle as pickle

mfc = None
def readMelLogSpec( feat_dir ):
    global mfc
    n_pad = 5
    feats = []
    num = 0
    f_dict = {}
    for d, ds, f in os.walk(feat_dir):
        for fname in f:
            m = re.search( r"(.*)\.mfc", fname )
            if( m ): #Check if file has .mfc extension
                file = m.group(1)
                num += 1
                f = open(d+'/'+fname, "r")
                l = f.read().split('\n')
                if( l[len(l)-1] == '' ): l.pop()
                l = [ list(i.split(' ')) for i in l ]
                z = [ [ float(0) for j in range(0,40) ] for i in range(0,n_pad) ] #Zero padding
                l = [ [float(j) for j in i] for i in l ] #Make strings into floats
                l = z+l+z
                n = []
                for i in range(n_pad,len(l)-n_pad):
                    m = []
                    for j in l[i-n_pad:i+(n_pad+1)]:
                        m+=j
                    n.append(m)
                n = np.asarray( n, dtype=float )
                f_dict[file] = n
    
    #pickle.dump( f_dict, open( 'mls.p', 'w' ) )
    mfc = f_dict
    return f_dict


if __name__ == '__main__':
    readMelLogSpec( '../an4/feat_dir/an4_clstk' )
