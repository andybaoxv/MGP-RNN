#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 5 Jun 2017.

Fit MGP-RNN model on full data, with Lanczos and CG to speed things up.

@author: josephfutoma
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
from scipy.sparse import csr_matrix
    
from util import pad_rawdata,SE_kernel,OU_kernel,dot,CG,Lanczos,block_CG,block_Lanczos

#####
##### Numpy functions used to simulate data and pad raw data to feed into TF
#####

def sim_multitask_GP(N,M,L_f,noise_vars,length,trainfrac):
    """
    draw from a multitask GP.  
    
    we continue to assume for now that the dim of the input space is 1, ie just time

    N: number of inputs (assumed that each input is dim 1)
    M: number of tasks
    
    train_frac: proportion of full M x N data matrix Y to include

    """
    n = N*M
    x = np.sort(np.random.uniform(0,10.0,N))    
    
    K_x = OU_kernel_np(length,x) #just a correlation function
    K_f = np.dot(L_f,L_f.T)                    
    Sigma = np.diag(noise_vars)

    K = np.kron(K_f,K_x) + np.kron(Sigma,np.eye(N)) + 1e-6*np.eye(n)
    L_K = np.linalg.cholesky(K)
    
    y = np.dot(L_K,np.random.normal(0,1,n))
    
    ind_kf = np.tile(np.arange(M),(N,1)).flatten('F') #vec by column
    ind_kx = np.tile(np.arange(N),(M,1)).flatten()
               
    perm = np.random.permutation(N*M)
    n_train = int(trainfrac*N*M)
    train_inds = perm[:n_train]
    
    y_ = y[train_inds]
    ind_kf_ = ind_kf[train_inds]
    ind_kx_ = ind_kx[train_inds]
    
    return y_,x,ind_kf_,ind_kx_

def OU_kernel_np(length,x):
    """ just a correlation function, for identifiability 
    """
    x1 = np.reshape(x,[-1,1]) #colvec
    x2 = np.reshape(x,[1,-1]) #rowvec
    K_xx = np.exp(-np.abs(x1-x2)/length)    
    return K_xx

#####
##### Tensorflow functions 
#####

def draw_GP(length,noises,Lf,Kf,Yi,Ti,Xi,ind_kfi,ind_kti):
    """ 
    given GP hyperparams and data values at observation times, draw from 
    conditional GP
    
    inputs:
        length,noises,Lf,Kf: GP params
        Yi: observation values
        Ti: observation times
        Xi: grid points (new times for rnn)
        ind_kfi,ind_kti: indices into Y
    returns:
        Mu, L_Sigma: normal parameters for p(f(X)|Y,T,W,params) 
    """  
    ny = tf.shape(Yi)[0]
    K_tt = OU_kernel(length,Ti,Ti)
    D = tf.diag(noises)

    grid_f = tf.meshgrid(ind_kfi,ind_kfi) #same as np.meshgrid
    Kf_big = tf.gather_nd(Kf,tf.stack((grid_f[0],grid_f[1]),-1))
    
    grid_t = tf.meshgrid(ind_kti,ind_kti) 
    Kt_big = tf.gather_nd(K_tt,tf.stack((grid_t[0],grid_t[1]),-1))

    Kf_Ktt = tf.multiply(Kf_big,Kt_big)

    DI_big = tf.gather_nd(D,tf.stack((grid_f[0],grid_f[1]),-1))
    DI = tf.diag(tf.diag_part(DI_big)) #D kron I
    
    #data covariance. cholesky no longer! use CG for matrix-vec products
    Ky = Kf_Ktt + DI + 1e-6*tf.eye(ny)   

    ### build out cross-covariances and covariance at grid
    
    nx = tf.shape(Xi)[0]
    
    K_xx = OU_kernel(length,Xi,Xi)
    K_xt = OU_kernel(length,Xi,Ti)
                       
    ind = tf.concat([tf.tile([i],[nx]) for i in range(M)],0)
    grid = tf.meshgrid(ind,ind)
    Kf_big = tf.gather_nd(Kf,tf.stack((grid[0],grid[1]),-1))
    ind2 = tf.tile(tf.range(nx),[M])
    grid2 = tf.meshgrid(ind2,ind2)
    Kxx_big =  tf.gather_nd(K_xx,tf.stack((grid2[0],grid2[1]),-1))
    
    K_ff = tf.multiply(Kf_big,Kxx_big) #cov at grid points           
                 
    full_f = tf.concat([tf.tile([i],[nx]) for i in range(M)],0)            
    grid_1 = tf.meshgrid(full_f,ind_kfi,indexing='ij')
    Kf_big = tf.gather_nd(Kf,tf.stack((grid_1[0],grid_1[1]),-1))
    full_x = tf.tile(tf.range(nx),[M])
    grid_2 = tf.meshgrid(full_x,ind_kti,indexing='ij')
    Kxt_big = tf.gather_nd(K_xt,tf.stack((grid_2[0],grid_2[1]),-1))

    K_fy = tf.multiply(Kf_big,Kxt_big)
       
    #now get draws!
    y_ = tf.reshape(Yi,[-1,1])
    Mu = tf.matmul(K_fy,CG(Ky,y_))
        
    #Never need to explicitly compute Sigma! Just need matrix-vector products in Lanczos algorithm
    def Sigma_mul(vec):
        """ vec must be a column vector, shape (?,1) """
        return tf.matmul(K_ff,vec) - tf.matmul(K_fy,CG(Ky,tf.matmul(tf.transpose(K_fy),vec)))
            
    xi = tf.random_normal((nx*M,1))
    draw = Mu + Lanczos(Sigma_mul,nx*M,xi)
    draw_reshape = tf.transpose(tf.reshape(draw,[1,M,nx]),perm=[0,2,1]) #reshape goes by row; we want by column
              
    return draw_reshape    
        
def get_GP_samples(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
                   num_rnn_grid_times,med_cov_grid):
    """
    returns samples from GP at evenly-spaced gridpoints
    """ 
    grid_max = tf.shape(X)[1]
    Z = tf.zeros([0,grid_max,input_dim])
    
    N = tf.shape(T)[0] #number of observations
        
    #setup tf while loop (have to use this bc loop size is variable)
    def cond(i,Z):
        return i<N
    
    def body(i,Z):
        Yi = tf.reshape(tf.slice(Y,[i,0],[1,num_obs_values[i]]),[-1])
        Ti = tf.reshape(tf.slice(T,[i,0],[1,num_obs_times[i]]),[-1])
        ind_kfi = tf.reshape(tf.slice(ind_kf,[i,0],[1,num_obs_values[i]]),[-1])
        ind_kti = tf.reshape(tf.slice(ind_kt,[i,0],[1,num_obs_values[i]]),[-1])
        Xi = tf.reshape(tf.slice(X,[i,0],[1,num_rnn_grid_times[i]]),[-1])
        X_len = num_rnn_grid_times[i]
                
        GP_draws = draw_GP(Yi,Ti,Xi,ind_kfi,ind_kti)
        pad_len = grid_max-X_len #pad by this much
        padded_GP_draws = tf.concat([GP_draws,tf.zeros((n_mc_smps,pad_len,M))],1) 

        medcovs = tf.slice(med_cov_grid,[i,0,0],[1,-1,-1])
        tiled_medcovs = tf.tile(medcovs,[n_mc_smps,1,1])
        padded_GPdraws_medcovs = tf.concat([padded_GP_draws,tiled_medcovs],2)
        
        Z = tf.concat([Z,padded_GPdraws_medcovs],0)        

        return i+1,Z  
    
    i = tf.constant(0)
    i,Z = tf.while_loop(cond,body,loop_vars=[i,Z],
                shape_invariants=[i.get_shape(),tf.TensorShape([None,None,None])])

    return Z

def get_preds(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
              num_rnn_grid_times,med_cov_grid):
    """
    helper function. takes in (padded) raw datas, samples MGP for each observation, 
    then feeds it all through the LSTM to get predictions.
    
    inputs:
        Y: array of observation values (labs/vitals). batchsize x batch_maxlen_y
        T: array of observation times (times during encounter). batchsize x batch_maxlen_t
        ind_kf: indiceste into each row of Y, pointing towards which lab/vital. same size as Y
        ind_kt: indices into each row of Y, pointing towards which time. same size as Y
        num_obs_times: number of times observed for each encounter; how long each row of T really is
        num_obs_values: number of lab values observed per encounter; how long each row of Y really is
        num_rnn_grid_times: length of even spaced RNN grid per encounter
                    
    returns:
        predictions (unnormalized log probabilities) for each MC sample of each obs
    """
    Z = get_GP_samples(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
                       num_rnn_grid_times,med_cov_grid) #batchsize*num_MC x batch_maxseqlen x num_inputs  
    Z.set_shape([None,None,input_dim]) #somehow lost shape info, but need this
       
    N = tf.shape(T)[0] #number of observations 
    init_state = ( tf.contrib.rnn.LSTMStateTuple(tf.tile(tf.slice(init_cs,[0,0],[1,-1]),[N,1]),
                        tf.tile(tf.slice(init_hs,[0,0],[1,-1]),[N,1])), 
                   tf.contrib.rnn.LSTMStateTuple(tf.tile(tf.slice(init_cs,[1,0],[1,-1]),[N,1]),
                        tf.tile(tf.slice(init_hs,[1,0],[1,-1]),[N,1])) )
    
    outputs, states = tf.nn.dynamic_rnn(cell=stacked_lstm,inputs=Z,
                                        initial_state=init_state,
                                        dtype=tf.float32,
                                        sequence_length=num_rnn_grid_times)
    
    final_outputs = states[n_layers-1][1]
    preds =  tf.matmul(final_outputs, out_weights) + out_biases  
    return preds

def get_probs_and_accuracy(preds,O):
    """
    helper function. we have a prediction for each MC sample of each observation
    in this batch.  need to distill the multiple preds from each MC into a single
    pred for this observation.  also get accuracy. use true probs to get ROC, PR curves in sklearn
    """
    probs = tf.exp(preds[:,1] - tf.reduce_logsumexp(preds, axis = 1)) #normalize; and drop a dim so only prob of positive case
           
    #compare to truth; just use cutoff of 0.5 for right now to get accuracy
    correct_pred = tf.equal(tf.cast(tf.greater(probs,0.5),tf.int32), O)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
    return probs,accuracy

if __name__ == "__main__":    
    seed = 8675309
    rs = np.random.RandomState(seed) #fixed seed in np
    
    
    #####
    ##### Setup ground truth and sim some data from a GP
    #####
    
    M = 10 #number of longitudinal variables
    n_meds = 5 #number of medications
    n_covs = 10  #number of baseline covariates
    num_obs = 10000 #number of simulated patient encounters

    #true_Kf = 0.8*np.ones((M,M))
    #true_Kf[np.diag_indices(M)] = 1.0 #diag scale elements
    tmp = np.random.normal(0,.5,(M,M))
    true_Kf = np.dot(tmp,tmp.T)
    
    true_Lf = np.linalg.cholesky(true_Kf)
    true_noises = np.linspace(.02,.08,M)
    true_length = 1.0
    
    train_frac = 0.08
    t = int(train_frac*N*M)

    ytr = np.zeros((num_obs,t))
    xtr = np.zeros((num_obs,N))
    ind_kf_tr = np.zeros((num_obs,t),dtype='int32')
    ind_kx_tr = np.zeros((num_obs,t),dtype='int32')

    for i in range(8500,num_obs):
        ytr[i,:],xtr[i,:],ind_kf_tr[i,:],ind_kx_tr[i,:] = sim_multitask_GP(N,M,true_Lf,true_noises,true_length,train_frac)  
    print("data simmed")
    
    
    
    (encIDs,covs,labels,times,values,ind_lvs,ind_times,meds,
            num_obs_times,num_obs_values,total_times,num_meds_admin,
            lab_names,med_names) = load_dat()
    N_tot = len(labels)
    
    train_test_perm = rs.permutation(N_tot)
    tr_frac = 0.8
    val_frac = 0.1
    te_frac = 0.1
    te_ind = train_test_perm[:int(val_frac*N_tot)]
    tr_ind = train_test_perm[int(val_frac*N_tot):]
    Nte = len(te_ind); Ntr = len(tr_ind)
    

    
    covs_tr = covs[tr_ind,:]; covs_te = covs[te_ind,:]
    labels_tr = labels[tr_ind]; labels_te = labels[te_ind]
    times_tr = times[tr_ind]; times_te = times[te_ind]
    values_tr = values[tr_ind]; values_te = values[te_ind]
    ind_lvs_tr = ind_lvs[tr_ind]; ind_lvs_te = ind_lvs[te_ind]
    ind_times_tr = ind_times[tr_ind]; ind_times_te = ind_times[te_ind]
    meds_tr = meds[tr_ind]; meds_te = meds[te_ind]
    num_obs_times_tr = num_obs_times[tr_ind]; num_obs_times_te = num_obs_times[te_ind]
    num_obs_values_tr = num_obs_values[tr_ind]; num_obs_values_te = num_obs_values[te_ind]
    total_times_tr = total_times[tr_ind]; total_times_te = total_times[te_ind]; 
                                
    #now get the rnn grid times.
    grid_size = 4 #1.0 hours
    grid_hours = 1.0 #spacing between grid points, in hours
    
    rnn_grid_times_tr = []; 
    for i in range(Ntr):
       last_t = total_times_tr[i]/4.0-0.25 
       first_t = last_t % grid_hours
       rnn_grid_times_tr.append(np.arange(first_t,last_t+grid_hours,grid_hours))
    rnn_grid_times_tr = np.array(rnn_grid_times_tr)
    num_rnn_grid_times_tr = np.array([len(x) for x in rnn_grid_times_tr])

    rnn_grid_times_te = []; 
    for i in range(Nte):
       last_t = total_times_te[i]/4.0-0.25 
       first_t = last_t % grid_hours
       rnn_grid_times_te.append(np.arange(first_t,last_t+grid_hours,grid_hours))
    rnn_grid_times_te = np.array(rnn_grid_times_te)
    num_rnn_grid_times_te = np.array([len(x) for x in rnn_grid_times_te])

    #ok, now get the meds at each grid point to feed into rnn
    #convert to dense for ease
    
    #TODO: this is going to be different for med effect; don't want to simply collapse med times
    meds_on_grid_tr = []
    for i in range(Ntr):
        dense_med = meds_tr[i].toarray()
        this_grid = []
        tmp = np.zeros(8)
        grid_pt = rnn_grid_times_tr[i][0]
        t = 0.0
        for j in range(dense_med.shape[0]):
            tmp = tmp + dense_med[j,:]
            if t==grid_pt:
                this_grid.append(tmp)
                tmp = np.zeros(8)
                grid_pt += grid_hours
            t += 0.25
        meds_on_grid_tr.append(np.array(this_grid))
    meds_on_grid_tr = np.array(meds_on_grid_tr)

    meds_on_grid_te = []
    for i in range(Nte):
        dense_med = meds_te[i].toarray()
        this_grid = []
        tmp = np.zeros(8)
        grid_pt = rnn_grid_times_te[i][0]
        t = 0.0
        for j in range(dense_med.shape[0]):
            tmp = tmp + dense_med[j,:]
            if t==grid_pt:
                this_grid.append(tmp)
                tmp = np.zeros(8)
                grid_pt += grid_hours
            t += 0.25
        meds_on_grid_te.append(np.array(this_grid))
    meds_on_grid_te = np.array(meds_on_grid_te)

    print("data fully setup!")    
    sys.stdout.flush()

    #####
    ##### Setup model and graph
    ##### 
    
    # Learning Parameters
    learning_rate = 0.002
    L2_penalty = 1e-3
    training_iters = 10000 #num epochs
    batch_size = 100 
    test_freq = 100 #eval on test set after this many batches
    
    # Network Parameters
    n_hidden = 64 # hidden layer num of features; assumed same
    n_layers = 2 # number of layers of stacked LSTMs
    n_classes = 2 #binary outcome
    input_dim = M+num_meds+num_covs #dimensionality of input sequence.
    
    # Create graph
    ops.reset_default_graph()
    sess = tf.Session()      
    
    ##### tf Graph - inputs 
    
    #observed values, times, inducing times; padded to longest in the batch
    Y = tf.placeholder("float", [None,None]) #batchsize x batch_maxdata_length
    T = tf.placeholder("float", [None,None]) #batchsize x batch_maxdata_length
    ind_kf = tf.placeholder(tf.int32, [None,None]) #index tasks in Y vector
    ind_kt = tf.placeholder(tf.int32, [None,None]) #index inputs in Y vector
    X = tf.placeholder("float", [None,None]) #grid points. batchsize x batch_maxgridlen
    med_cov_grid = tf.placeholder("float", [None,None,num_meds+num_covs]) #combine w GP smps to feed into RNN
    
    O = tf.placeholder(tf.int32, [None]) #labels. input is NOT as one-hot encoding; convert at each iter
    num_obs_times = tf.placeholder(tf.int32, [None]) #number of observation times per encounter 
    num_obs_values = tf.placeholder(tf.int32, [None]) #number of observation values per encounter 
    num_rnn_grid_times = tf.placeholder(tf.int32, [None]) #length of each grid to be fed into RNN in batch
    
    N = tf.shape(Y)[0]                         
                                                                                                                                                                                      
    #also make O one-hot encoding, for the loss function
    O_onehot = tf.one_hot(O,n_classes)

    ##### tf Graph - variables to learn
        
    ### GP parameters (unconstrained)
    
    #in fully separable case all tasks share same x-covariance
    log_length = tf.Variable(tf.random_normal([1],mean=1,stddev=0.1),name="GP-log-length") 
    length = tf.exp(log_length)
    
    #different noise level of each task
    log_noises = tf.Variable(tf.random_normal([M],mean=-2,stddev=0.1),name="GP-log-noises")
    noises = tf.exp(log_noises)
    
    #init cov between tasks. we'll optimize some low rank approx    
    L_f_init = tf.Variable(tf.eye(M),name="GP-Lf")
    Lf = tf.matrix_band_part(L_f_init,-1,0)
    Kf = tf.matmul(Lf,tf.transpose(Lf))

    ### RNN params

    # Create network
    
    #learn initial RNN state variables
    init_cs = tf.Variable(tf.random_normal([n_layers,n_hidden],mean=0,stddev=0.05),name="RNN-init_cs")
    init_hs = tf.Variable(tf.random_normal([n_layers,n_hidden],mean=0,stddev=0.05),name="RNN-init_hs")
    
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_hidden) for _ in range(n_layers)])

    # Weights at the last layer given deep LSTM output
    out_weights = tf.Variable(tf.random_normal([n_hidden, n_classes],stddev=0.1),name="Softmax/W")
    out_biases = tf.Variable(tf.random_normal([n_classes],stddev=0.1),name="Softmax/b")

    ##### Get predictions and feed into optimization
    preds = get_preds(Y,T,X,ind_kf,ind_kt,num_obs_times,
                      num_obs_values,num_rnn_grid_times,med_cov_grid)    
    probs,accuracy = get_probs_and_accuracy(preds,O)
    
    # Define optimization problem
    loss_fit = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=preds,labels=O_onehot))
    with tf.variable_scope("",reuse=True):
        loss_reg = L2_penalty*(tf.reduce_sum(tf.square(out_weights)) + 
                           tf.reduce_sum(tf.square(tf.get_variable('rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights'))) +
                           tf.reduce_sum(tf.square(tf.get_variable('rnn/multi_rnn_cell/cell_1/basic_lstm_cell/weights'))))
    loss = loss_fit + loss_reg                  
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
   
    ##### Initialize globals and get ready to start!
    sess.run(tf.global_variables_initializer())
     
    #setup minibatch indices
    starts = np.arange(0,Ntr,batch_size)
    ends = np.arange(batch_size,Ntr+1,batch_size)
    if ends[-1]<Ntr: 
        ends = np.append(ends,Ntr)
    num_batches = len(ends) 
    
    T_pad_te,Y_pad_te,ind_kf_pad_te,ind_kt_pad_te,X_pad_te,meds_cov_pad_te = pad_rawdata(
                    times_te,values_te,ind_lvs_te,ind_times_te,
                    rnn_grid_times_te,meds_on_grid_te,covs_te) 
    
    ##### Main training loop
    saver = tf.train.Saver(max_to_keep = None)
    saver.restore(sess,"model_checkpoints/MGP-RNN-mean/-600") #warm start to mean-MGP

    total_batches = 0
    for i in range(training_iters):
        #train
        epoch_start = time()
        print("Starting epoch "+"{:d}".format(i))
        perm = rs.permutation(Ntr)
        batch = 0 
        for s,e in zip(starts,ends):
            batch_start = time()
            inds = perm[s:e]
            T_pad,Y_pad,ind_kf_pad,ind_kt_pad,X_pad,meds_cov_pad = pad_rawdata(
                    times_tr[inds],values_tr[inds],ind_lvs_tr[inds],ind_times_tr[inds],
                    rnn_grid_times_tr[inds],meds_on_grid_tr[inds],covs_tr[inds,:]) 
            sess.run(tf.local_variables_initializer(),feed_dict={X:X_pad})  #reinit!!                                      
            feed_dict={Y:Y_pad,T:T_pad,ind_kf:ind_kf_pad,ind_kt:ind_kt_pad,X:X_pad,
               med_cov_grid:meds_cov_pad,num_obs_times:num_obs_times_tr[inds],
               num_obs_values:num_obs_values_tr[inds],
               num_rnn_grid_times:num_rnn_grid_times_tr[inds],O:labels_tr[inds]}                         
                     
            loss_,_ = sess.run([loss,train_op],feed_dict)
            
            print("Batch "+"{:d}".format(batch)+"/"+"{:d}".format(num_batches)+\
                  ", took: "+"{:.3f}".format(time()-batch_start)+", loss: "+"{:.5f}".format(loss_))
            sys.stdout.flush()
            batch += 1; total_batches += 1

            if total_batches % test_freq == 0: #check test set every so often
                #out of sample performance                
                hours_back = np.array([0,4,8])
                
                for j in range(3):
                    test_t = time()

                    hour = hours_back[j]
                    grid_min = hour+1
                
                    inds = np.where(num_rnn_grid_times_te>=grid_min)[0]
                    sess.run(tf.local_variables_initializer(),feed_dict={X:X_pad_te[inds]})  #reinit!! 
                    feed_dict={Y:Y_pad_te[inds,:],T:T_pad_te[inds,:],ind_kf:ind_kf_pad_te[inds,:],ind_kt:ind_kt_pad_te[inds,:],X:X_pad_te[inds,:],
                           med_cov_grid:meds_cov_pad_te[inds,:,:],num_obs_times:num_obs_times_te[inds],
                           num_obs_values:num_obs_values_te[inds],
                           num_rnn_grid_times:num_rnn_grid_times_te[inds]-hour,O:labels_te[inds]}                               
                    te_probs,te_acc,te_loss = sess.run([probs,accuracy,loss],feed_dict)     
                    te_auc = roc_auc_score(labels_te[inds], te_probs)
                    te_prc = average_precision_score(labels_te[inds], te_probs)   
                  
                    print("Epoch "+str(i)+", seen "+str(total_batches)+" total batches. Testing Took "+\
                          "{:.2f}".format(time()-test_t)+\
                          ". OOS, "+str(hour)+" hours back: Loss: "+"{:.5f}".format(te_loss)+ \
                          " Acc: "+"{:.5f}".format(te_acc)+", AUC: "+ \
                          "{:.5f}".format(te_auc)+", PRC: "+"{:.5f}".format(te_prc))
                    sys.stdout.flush()    
            
                saver.save(sess, "model_checkpoints/MGP-RNN/", global_step=total_batches)
        print("Finishing epoch "+"{:d}".format(i)+", took "+\
              "{:.3f}".format(time()-epoch_start))     