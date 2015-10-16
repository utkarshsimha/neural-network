import os
import re
import numpy as np
import cPickle as pickle
import time

senone = None
def read_segmentation(directory):
	global senone
	d={}	
	#lines = 0
	n_out = 0
	out_vec=[]
	for root,dirs,file in list(os.walk(directory)):
		for files in file:
			m =  re.match(r"(.*)\.stsegid",files)
			if m:
				#print m.group(1)
				with open(root+"/"+files,"r") as f:
					n_out=int(f.readline().strip('\n'))
					f_vec=[]
					count=0
					f.seek(0,0)
					for i in f.readlines():
						if count == 0:
							count=1
						else:
							l=[0]*n_out
							count+=1
							i=i.strip('\n')
							l[int(i)]=1
							f_vec.append(l)
					d[m.group(1)]=np.asarray(f_vec)
	#pickle.dump(d,open("senid.p","wb"))
	senone = d
	return d
	#print n_out

if __name__ == '__main__':
    read_segmentation('seg')
