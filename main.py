from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import pdb
import time
import math
import pickle
import random
import numpy as np
#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.logging.set_verbosity(tf.logging.INFO)

from model import UWMMSE
#from channel import *

# Experiment 
dataID = sys.argv[1]
exp = sys.argv[2]
if len( sys.argv ) > 3:
    mode = sys.argv[3]

dataPath = 'data/'+dataID+'/'
modelPath = 'models/'+dataID+'/'
resultPath = 'results/'+dataID+'/'

# Maximum available power at each node
Pmax = 1.0

# Noise power
var_db = -91
var = 10**(var_db/10)

# Features
feature_dim = 2  # U and V

# Batch size
batch_size = 64

# Train iterations
tr_iters = 15000

# Test samples
te_smpls = 10048

# Layers UWMMSE = 4 (default)  WMMSE = 100 (default)
layers = 4 if exp == 'uwmmse' else 100

# Learning rate
learning_rate=1e-2

# Number of epochs
nEpoch = 100

# Pickle Load
def pload( path ):
    dump = pickle.load( open( path, 'rb' ) )
    return( dump )
    
# Pickle dump
def pdump( dump, path ):
    f = open(path,'wb')
    pickle.dump(dump, f)
    f.close()
    
# Build random geometric graph
def build_adhoc_network( nNodes, r=1, pl=2.2 ):
    transmitters = np.random.uniform(low=-nNodes/r, high=nNodes/r, size=(nNodes,2))
    receivers = transmitters + np.random.uniform(low=-nNodes/4,high=nNodes/4, size=(nNodes,2))

    L = np.zeros((nNodes,nNodes))

    for i in np.arange(nNodes):
        for j in np.arange(nNodes):
            d = np.linalg.norm(transmitters[i,:]-receivers[j,:])
            L[i,j] = np.power(d,-pl)

    return( dict(zip(['tx', 'rx'],[transmitters, receivers] )), L )

# Simuate Fading
def sample_graph(A, nNodes, alpha=1.):
    samples = np.random.rayleigh(alpha, (nNodes, nNodes))
    #samples = (samples + np.transpose(samples,(1,0)))/2

    PP = samples[None,:,:] * A
    return PP[0]

# Generate data
def genTeData(usrs):
    tS = []
    tH = []
    K = random.sample(usrs,1)[0]
    _, A = build_adhoc_network( K )
    for i in range(int(te_smpls/batch_size)):
        #K = random.sample(usrs,1)[0]
        tS.append(K)
        for j in range(batch_size):
            #_, A = build_adhoc_network( K )
            tH.append(sample_graph(A, K))
        
    return(A,tS,tH)
    #return(tS,tH)

def genTrData(A, K):
    H = []
    for j in range(batch_size):
        #_, A = build_adhoc_network( K )
        H.append(sample_graph(A, K))
    
    return( np.asarray(H) )

# Number of variables
def num_var():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters
    
    return(total_parameters)
    
# Create Model Instance
def create_model( session, exp='uwmmse' ):
    # Create
    model = UWMMSE( Pmax=Pmax, var=var, feature_dim=feature_dim, batch_size=batch_size, layers=layers, learning_rate=learning_rate, exp=exp )
    
    if exp == 'uwmmse':
        print("UWMMSE model created with {} trainable parameters\n".format(num_var()))

    # Initialize variables ( To train from scratch )
    session.run(tf.compat.v1.global_variables_initializer())
    
    return model

# Train
def mainTrain():        
    # Create/Load dataset
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)
        usrs = list(range(20,21,1))
        A, test_sizes, test_H = genTeData(usrs)
        pdump( A, dataPath+'A.pkl' )
        pdump( test_sizes, dataPath+'test_sizes.pkl' )
        pdump( test_H, dataPath+'test_H.pkl' )
        #pdump( val_H, dataPath+'val_H.pkl' )
        print("Created dataset")
    else:
        test_sizes = pload( dataPath+'test_sizes.pkl' )
        A = pload( dataPath+'A.pkl' )
        test_H = pload( dataPath+'test_H.pkl' )
        #val_H = pload( dataPath+'val_H.pkl' )
        print("Loaded dataset")
        
    #test_H = pload( dataPath+'test_H.pkl' )
    
    # Initiate TF session for WMMSE
    with tf.compat.v1.Session(config=config) as sess:
        # WMMSE experiment
        if exp == 'wmmse':
        
            # Create model 
            model = create_model( sess, exp )
            
            # Test
            test_iter = te_smpls
                    
            print( '\nWMMSE Started\n' )

            t = 0.
            test_rate = 0.0
            sum_rate = []
            power = []

            for batch in range(0,test_iter,batch_size):
                batch_test_inputs = np.asarray(test_H[batch:batch+batch_size])
                batch_test_inputs_ = batch_test_inputs + np.random.normal(0.0,0.0025,batch_test_inputs.shape) 
                start = time.time()
                
                #### Replace batch_test_inputs_ with batch_test_inputs for evaluating on unperturbed inputs
                avg_rate, batch_rate1, batch_power, batch_rate, _, _ = model.eval( sess, inputs=batch_test_inputs_, inputs_=batch_test_inputs)
                #print(avg_rate);pdb.set_trace()
                t += (time.time() - start)
                test_rate += -avg_rate
                sum_rate.append( batch_rate )
                power.append(batch_power)
            
            test_rate /= test_iter
            test_rate *= batch_size

            # Average per-iteration test time
            t = t / test_iter
            t *= batch_size
            
            log = "Test_rate = {:.3f}, Time = {:.3f} sec\n"
            print(log.format( test_rate, t))    
            
            if not os.path.exists(resultPath):
                os.makedirs(resultPath)
            #### Remove '_e' when evauating without perturbation
            if np.unique(test_sizes).shape[0] > 1:
                mean_sum_rate, sizes = proc_res(sum_rate, test_sizes)
                pdump( mean_sum_rate, resultPath+'wmmse_rate.pkl' )
                pdump( sizes, resultPath+'sizes.pkl' )
            else:
                mean_sum_rate = np.mean( sum_rate, axis=1 )
                sum_rate = np.concatenate( sum_rate, axis=0 )
                pdump( sum_rate, resultPath+'wmmse_rate_e.pkl' )
                #pdump( fi, resultPath+'wmmse_fi.pkl' )
                
            pdump( power, resultPath+'wmmse_power_e.pkl' )
        else:
            # Create model 
            model = create_model( sess, exp )

            if mode == 'train':
                # Create model path
                if not os.path.exists(modelPath):
                    os.makedirs(modelPath)
                #else:
                    ## Restore best saved model
                #model.restore(sess, path=modelPath)
                max_rate = 0.
                train_iter = 15000#30000#
                train_H = []
                
                usrs = list(range(20,21,1))

                #Training loop
                print( '\nUWMMSE Training Started\n' )
                
                start = time.time()
                train_rate = 0.0
                rng = np.random.RandomState(7134)
                for it in range(train_iter):
                    usr = random.sample(usrs,1)[0]
                    batch_train_inputs = genTrData(A,usr)
                    step_rate, batch_rate, power = model.train( sess, inputs=batch_train_inputs)
                    if np.isnan(step_rate) or np.isinf(step_rate) :
                        pdb.set_trace()
                    
                    train_rate += -step_rate
                
                    if ( ( (it+1) % 500 ) == 0):
                        train_rate /= 500
                                            
                        # Validate
                        val_rate = 0.0
                        val_iter = te_smpls#100
                        
                        for batch in range(0,val_iter,batch_size):
                            batch_val_inputs = np.asarray(test_H[batch:batch+batch_size])
                            avg_rate, batch_rate, batch_power, _, _, _ = model.eval( sess, inputs=batch_val_inputs, inputs_=batch_val_inputs)
                            if np.isnan(avg_rate) or np.isinf(avg_rate):
                                pdb.set_trace()
                            val_rate += -avg_rate
                            
                        val_rate /= val_iter
                        val_rate *= batch_size
                        log = "Iters {}/{}, Train Sum_rate = {:.3f}, \nValid Sum_rate = {:.3f}, Time Elapsed = {:.3f} sec\n"
                        print(log.format( it+1, train_iter, train_rate, val_rate, time.time() - start) )
                                            
                        train_rate = 0.0

                        if (val_rate > max_rate):
                            max_rate = val_rate
                            model.save(sess, path=modelPath+'uwmmse-model', global_step=(it+1))

                print( 'Training Complete' )

            # Test
            t = 0.
            test_rate = 0.0
            test_iter = te_smpls
            
            power = []
            sum_rate = []
            vlist = []
            alist = []
            ulist = []
            wlist = []
            # Restore best saved model
            model.restore(sess, path=modelPath)

            print( '\nUWMMSE Testing Started\n' )

            for batch in range(0,test_iter,batch_size):
                batch_test_inputs = np.asarray(test_H[batch:batch+batch_size])
                batch_test_inputs_ = batch_test_inputs + np.random.normal(0.0,0.0025,batch_test_inputs.shape) 
                start = time.time()
                
                #### Replace batch_test_inputs_ with batch_test_inputs for evaluating on unperturbed inputs
                avg_rate, batch_rate1, batch_power, batch_rate, vv, aa, uu, ww = model.eval( sess, inputs=batch_test_inputs_, inputs_=batch_test_inputs)
                vlist.append(vv)
                ulist.append(uu)
                wlist.append(ww)
                alist.append(aa)
                #pdb.set_trace()
                t += (time.time() - start)
                if np.isnan(avg_rate) or np.isinf(avg_rate):
                    pdb.set_trace()
                sum_rate.append( batch_rate )
                power.append( batch_power )
                test_rate += -avg_rate
            
            #### Remove '_e' when evauating without perturbation
            pdump(vlist,resultPath+'vlist_e.pkl')
            pdump(ulist,resultPath+'ulist_e.pkl')
            pdump(wlist,resultPath+'wlist_e.pkl')
            pdump(alist,resultPath+'alist_e.pkl')
            test_rate /= test_iter
            test_rate *= batch_size
            
            ## Average per-iteration test time   
            t = t / test_iter
            t *= batch_size
            
            log = "Test Sum_rate = {:.3f}, Time = {:.3f} sec\n"
            print(log.format( test_rate, t))
            
            if not os.path.exists(resultPath):
                os.makedirs(resultPath)
            
            if np.unique(test_sizes).shape[0] > 1:
                mean_sum_rates, sizes = proc_res(sum_rate,test_sizes)
                pdump( mean_sum_rates, resultPath+'uwmmse_rate.pkl' )
                pdump( sizes, resultPath+'sizes.pkl' )
            else:
                mean_sum_rate = np.mean( sum_rate, axis=1 )
                sum_rate = np.concatenate( sum_rate, axis=0 )
                pdump( sum_rate, resultPath+'uwmmse_rate_e.pkl' )
                #pdump( fi, resultPath+'uwmmse_fi.pkl' )
                
            pdump( power, resultPath+'uwmmse_power_e.pkl' )
            
if __name__ == "__main__":        
    import sys

    rn = np.random.randint(2**20)
    rn1 = np.random.randint(2**20)
    tf.set_random_seed(rn)
    np.random.seed(rn1)

    mainTrain()
