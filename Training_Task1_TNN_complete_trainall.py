from keras.layers import Input, Conv2D, Lambda, concatenate, Dense, Flatten,MaxPooling2D,Activation
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from sklearn.preprocessing import OneHotEncoder
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import random
import numpy as np
from sklearn.metrics import roc_curve
from itertools import permutations

adam_optim = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

def generate_triplet(x,y,testsize=0.3,ap_pairs=10,an_pairs=10):
    data_xy = tuple([x,y])

    trainsize = 1-testsize

    triplet_train_pairs = []
    triplet_test_pairs = []
    for data_class in sorted(set(data_xy[1])):

        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where(data_xy[1] != data_class)[0]       
        if same_class_idx.shape[0] < 15:
            same_class_sampleer_idx = random.choice(range(len(same_class_idx)))
            A_P_pairs = random.sample(list(permutations(same_class_idx,2)),k=ap_pairs) #Generating Anchor-Positive pairs
        else:
            same_class_sampleer_idx = random.choice(range(len(same_class_idx)-21))
            A_P_pairs = random.sample(list(permutations(same_class_idx[same_class_sampleer_idx:same_class_sampleer_idx+21],2)),k=ap_pairs) #Generating Anchor-Positive pairs
        Neg_idx = random.sample(list(diff_class_idx),k=an_pairs)
        

        #train
        A_P_len = len(A_P_pairs)
        Neg_len = len(Neg_idx)
        for ap in A_P_pairs[:int(A_P_len*trainsize)]:
            Anchor = data_xy[0][ap[0]]
            Positive = data_xy[0][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                triplet_train_pairs.append([Anchor,Positive,Negative])               
        #test
        for ap in A_P_pairs[int(A_P_len*trainsize):]:
            Anchor = data_xy[0][ap[0]]
            Positive = data_xy[0][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                triplet_test_pairs.append([Anchor,Positive,Negative])    
                
    return np.array(triplet_train_pairs), np.array(triplet_test_pairs)





# X_train = triplet_generation(trn_bl_ivector_10Classes.reshape(100,3,600), neg_class_num=20)

# X_train, X_test = triplet_generation(trn_bl_ivector.reshape(classes,3,600),neg_class_num=30)

## Triplet NN

def triplet_loss(y_true, y_pred, alpha = 5):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    total_lenght = y_pred.shape.as_list()[-1]
    anchor = y_pred[:,0:int(total_lenght*1/3)]
    positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
 
    return loss
def create_base_network(in_dims, out_dims):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Dense(600, input_shape=(in_dims,),activation='relu'))
    
    return model

anchor_input = Input((600, ), name='anchor_input')
positive_input = Input((600, ), name='positive_input')
negative_input = Input((600, ), name='negative_input')

# Shared embedding layer for positive and negative items
Shared_DNN = create_base_network(600,600)


encoded_anchor = Shared_DNN(anchor_input)
encoded_positive = Shared_DNN(positive_input)
encoded_negative = Shared_DNN(negative_input)

merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1)

model = Model(inputs=[anchor_input,positive_input, negative_input], outputs=merged_vector)
model.compile(loss=triplet_loss, optimizer=adam_optim)

model.summary()



## making dictionary to find blacklist pair between train and test dataset
# bl_match = np.loadtxt('data/bl_matching_dev.csv',dtype='string')
bl_match = np.loadtxt('data/bl_matching.csv',dtype='str')
dev2train={}
dev2id={}
train2dev={}
train2id={}
test2train={}
train2test={}
for iter, line in enumerate(bl_match):
    line_s = line.split(',')
    dev2train[line_s[1].split('_')[-1]]= line_s[3].split('_')[-1]
    dev2id[line_s[1].split('_')[-1]]= line_s[0].split('_')[-1]
    train2dev[line_s[3].split('_')[-1]]= line_s[1].split('_')[-1]
    train2id[line_s[3].split('_')[-1]]= line_s[0].split('_')[-1]
    test2train[line_s[2].split('_')[-1]]= line_s[3].split('_')[-1]
    train2test[line_s[3].split('_')[-1]]= line_s[2].split('_')[-1]

# import matplotlib
# matplotlib.use('Agg')



## helper functions

def load_ivector(filename):
    utt = np.loadtxt(filename,dtype='str',delimiter=',',skiprows=1,usecols=[0])
    ivector = np.loadtxt(filename,dtype='float32',delimiter=',',skiprows=1,usecols=range(1,601))
    spk_id = []
    for iter in range(len(utt)):
        spk_id = np.append(spk_id,utt[iter].split('_')[0])

    return spk_id, utt, ivector

def length_norm(mat):
# length normalization (l2 norm)
# input: mat = [utterances X vector dimension] ex) (float) 8631 X 600

    norm_mat = []
    for line in mat:
        temp = line/np.math.sqrt(sum(np.power(line,2)))
        norm_mat.append(temp)
    norm_mat = np.array(norm_mat)
    return norm_mat

def make_spkvec(mat, spk_label):
# calculating speaker mean vector
# input: mat = [utterances X vector dimension] ex) (float) 8631 X 600
#        spk_label = string vector ex) ['abce','cdgd']

#     for iter in range(len(spk_label)):
#         spk_label[iter] = spk_label[iter].split('_')[0]

    spk_label, spk_index  = np.unique(spk_label,return_inverse=True)
    spk_mean=[]
    mat = np.array(mat)

    # calculating speaker mean i-vector
    for i, spk in enumerate(spk_label):
        spk_mean.append(np.mean(mat[np.nonzero(spk_index==i)],axis=0))
    spk_mean = length_norm(spk_mean)
    return spk_mean, spk_label

def calculate_EER(trials, scores):
# calculating EER of Top-S detector
# input: trials = boolean(or int) vector, 1: postive(blacklist) 0: negative(background)
#        scores = float vector

    # Calculating EER
    fpr,tpr,threshold = roc_curve(trials,scores,pos_label=1)
    fnr = 1-tpr
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # print EER_threshold
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr+EER_fnr)
    
    print("Top S detector EER is %0.2f%%"% (EER*100))
    return EER

def get_trials_label_with_confusion(identified_label, groundtruth_label,dict4spk,is_trial ):
# determine if the test utterance would make confusion error
# input: identified_label = string vector, identified result of test utterance among multi-target from the detection system 
#        groundtruth_label = string vector, ground truth speaker labels of test utterances
#        dict4spk = dictionary, convert label to target set, ex) train2dev convert train id to dev id

    trials = np.zeros(len(identified_label))
    for iter in range(0,len(groundtruth_label)):
        enroll = identified_label[iter].split('_')[0]
        test = groundtruth_label[iter].split('_')[0]
        if is_trial[iter]:
            if enroll == dict4spk[test]:
                trials[iter]=1 # for Target trial (blacklist speaker)
            else:
                trials[iter]=-1 # for Target trial (backlist speaker), but fail on blacklist classifier
                
        else :
            trials[iter]=0 # for non-target (non-blacklist speaker)
    return trials


def calculate_EER_with_confusion(scores,trials):
# calculating EER of Top-1 detector
# input: trials = boolean(or int) vector, 1: postive(blacklist) 0: negative(background) -1: confusion(blacklist)
#        scores = float vector

    # exclude confusion error (trials==-1)
    scores_wo_confusion = scores[np.nonzero(trials!=-1)[0]]
    trials_wo_confusion = trials[np.nonzero(trials!=-1)[0]]

    # dev_trials contain labels of target. (target=1, non-target=0)
    fpr,tpr,threshold = roc_curve(trials_wo_confusion,scores_wo_confusion,pos_label=1, drop_intermediate=False)
    fnr = 1-tpr
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # EER withouth confusion error
    EER = fpr[np.argmin(np.absolute((fnr-fpr)))]
    
    # Add confusion error to false negative rate(Miss rate)
    total_negative = len(np.nonzero(np.array(trials_wo_confusion)==0)[0])
    total_positive = len(np.nonzero(np.array(trials_wo_confusion)==1)[0])
    fp= fpr*np.float(total_negative)  
    fn= fnr*np.float(total_positive) 
    fn += len(np.nonzero(trials==-1)[0])
    total_positive += len(np.nonzero(trials==-1)[0])
    fpr= fp/total_negative
    fnr= fn/total_positive

    # EER with confusion Error
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr+EER_fnr)
    
    print("Top 1 detector EER is %0.2f%% (Total confusion error is %d)"% ((EER*100), len(np.nonzero(trials==-1)[0])))
    return EER

# # Loading i-vector
# trn_bl_id, trn_bl_utt, trn_bl_ivector = load_ivector('data/trn_blacklist.csv')
# trn_bg_id, trn_bg_utt, trn_bg_ivector = load_ivector('data/trn_background.csv')
# dev_bl_id, dev_bl_utt, dev_bl_ivector = load_ivector('data/dev_blacklist.csv')
# dev_bg_id, dev_bg_utt, dev_bg_ivector = load_ivector('data/dev_background.csv')

trn_bl_ivector = pickle.load(open('./data/trn_bl_ivector','rb'))
trn_bg_ivector = pickle.load(open('./data/trn_bg_ivector','rb'))
dev_bl_ivector = pickle.load(open('./data/dev_bl_ivector','rb'))
dev_bg_ivector = pickle.load(open('./data/dev_bg_ivector','rb'))
trn_bl_id = pickle.load(open('./data/trn_bl_id','rb'))
trn_bg_id = pickle.load(open('./data/trn_bg_id','rb'))
dev_bl_id = pickle.load(open('./data/dev_bl_id','rb'))
dev_bg_id = pickle.load(open('./data/dev_bg_id','rb'))
trn_bl_utt = pickle.load(open('./data/trn_bl_utt','rb'))
trn_bg_utt = pickle.load(open('./data/trn_bg_utt','rb'))
dev_bl_utt = pickle.load(open('./data/dev_bl_utt','rb'))
dev_bg_utt = pickle.load(open('./data/dev_bg_utt','rb'))
tst_id = pickle.load(open('./data/tst_id','rb'))
test_utt = pickle.load(open('./data/test_utt','rb'))
tst_ivector = pickle.load(open('./data/tst_ivector','rb'))


# Calculating speaker mean vector
spk_mean, spk_mean_label = make_spkvec(trn_bl_ivector,trn_bl_id)

#length normalization
trn_bl_ivector = length_norm(trn_bl_ivector)
trn_bg_ivector = length_norm(trn_bg_ivector)
dev_bl_ivector = length_norm(dev_bl_ivector)
dev_bg_ivector = length_norm(dev_bg_ivector)
tst_ivector = length_norm(tst_ivector)

# load test set information
filename = 'data/tst_evaluation_keys.csv'
tst_info = np.loadtxt(filename,dtype='str',delimiter=',',skiprows=1,usecols=range(0,3))
tst_trials = []
tst_trials_label = []
tst_ground_truth =[]
for iter in range(len(tst_info)):
    tst_trials_label.extend([tst_info[iter,0]])
    if tst_info[iter,1]=='background':
        tst_trials = np.append(tst_trials,0)
        
    else:
        tst_trials = np.append(tst_trials,1)
        
# making trials of Dev set
dev_ivector = np.append(dev_bl_ivector, dev_bg_ivector,axis=0)
dev_trials = np.append( np.ones([len(dev_bl_id), 1]), np.zeros([len(dev_bg_id), 1]))

trn_ivector = np.append(trn_bl_ivector, trn_bg_ivector,axis=0)
trn_trials = np.append( np.ones([len(trn_bl_ivector), 1]), np.zeros([len(trn_bg_ivector), 1]))



# onehot = OneHotEncoder()
# trn_trials_onehot = onehot.fit_transform(trn_trials.reshape(-1,1)).toarray()
# dev_trails_onehot = onehot.fit_transform(dev_trials.reshape(-1,1)).toarray()
total_loss_history = []
total_valloss_history = []
err_ls = []
tst_EER1_ls = []
tst_EER2_ls = []

for i in range(101):    
    if i == 0:
        pass
    else:
        if err*100 < 1.5:
            model.load_weights(foldername + filename + '.hdf5')
        else:
            pass
            
        
    X_train, X_test = generate_triplet(np.append(trn_ivector,dev_ivector,axis=0),np.append(trn_trials,dev_trials,axis=0), ap_pairs=400, an_pairs=400,testsize=0.2)
    Anchor = X_train[:,0,:]
    Positive = X_train[:,1,:]
    Negative = X_train[:,2,:]
    Anchor_test = X_test[:,0,:]
    Positive_test = X_test[:,1,:]
    Negative_test = X_test[:,2,:]
    Y_dummy = np.empty((Anchor.shape[0],1))
    Y_dummy2 = np.empty((Anchor_test.shape[0],1))


    history = model.fit([Anchor,Positive,Negative],y=Y_dummy,validation_data=([Anchor_test,Positive_test,Negative_test],Y_dummy2), batch_size=1024, epochs=15)
    total_loss_history = total_loss_history + history.history['loss']
    total_valloss_history = total_valloss_history + history.history['val_loss']

    trained_model = Model(inputs=anchor_input, outputs=encoded_anchor)
    transformed_trn_bl_ivector = trained_model.predict(trn_bl_ivector)
    transformed_dev_ivector = trained_model.predict(dev_ivector)
    transformed_trn_ivector = trained_model.predict(trn_ivector)
    transformed_tst_ivector = trained_model.predict(tst_ivector)
    transformed_dev_bl_ivector = trained_model.predict(dev_bl_ivector)
    transformed_spk_mean, transformed_spk_mean_label = make_spkvec(transformed_trn_bl_ivector,trn_bl_id)

#################Report first score###################    
    scores = transformed_spk_mean.dot(transformed_dev_ivector.transpose())

    # Multi-target normalization
    blscores = transformed_spk_mean.dot(transformed_trn_bl_ivector.transpose())
    mnorm_mu = np.mean(blscores,axis=1)
    mnorm_std = np.std(blscores,axis=1)
    for iter in range(np.shape(scores)[1]):
        scores[:,iter]= (scores[:,iter] - mnorm_mu) / mnorm_std
    dev_scores = np.max(scores,axis=0)

    # Top-S detector EER
    print('\nDev set score using train set :')
    err = calculate_EER(dev_trials, dev_scores)
    
#################Report second score#####################  
    print('\nTest set score using train set:')  
    scores = transformed_spk_mean.dot(transformed_tst_ivector.transpose())
    # Multi-target normalization
    blscores = transformed_spk_mean.dot(transformed_trn_bl_ivector.transpose())
    mnorm_mu = np.mean(blscores,axis=1)
    mnorm_std = np.std(blscores,axis=1)
    for iter in range(np.shape(scores)[1]):
        scores[:,iter]= (scores[:,iter] - mnorm_mu) / mnorm_std
    tst_scores = np.max(scores,axis=0)
    tst_EER1 = calculate_EER(tst_trials, tst_scores)    
#################Report thrid score######################    
    print('\nTest set score using train + dev set:')
    # get dev set id consistent with Train set
    dev_bl_id_along_trnset = []
    for iter in range(len(dev_bl_id)):
        dev_bl_id_along_trnset.extend([dev2train[dev_bl_id[iter]]])

    # Calculating speaker mean vector
    # spk_mean, spk_mean_label = make_spkvec(np.append(trn_bl_ivector,dev_bl_ivector,0),np.append(trn_bl_id,dev_bl_id_along_trnset))
    transformed_spk_mean, transformed_spk_mean_label = make_spkvec(np.append(transformed_trn_ivector,transformed_dev_bl_ivector,0),np.append(trn_bl_id,dev_bl_id_along_trnset))

    #Cosine distance scoring on Test set
    scores = transformed_spk_mean.dot(transformed_tst_ivector.transpose())
    # # tst_scores = np.max(scores,axis=0)


    # # Multi-target normalization
    blscores = transformed_spk_mean.dot(np.append(transformed_trn_ivector.transpose(),transformed_dev_bl_ivector.transpose(),axis=1))
    mnorm_mu = np.mean(blscores,axis=1)
    mnorm_std = np.std(blscores,axis=1)
    for iter in range(np.shape(scores)[1]):
        scores[:,iter]= (scores[:,iter] - mnorm_mu) / mnorm_std
    tst_scores = np.max(scores,axis=0)

    # top-S detector EER

    tst_EER2 = calculate_EER(tst_trials, tst_scores)
    if err*100 < 1.9:
        try:
            foldername = './task1_trainall/'
            filename='task1_trainall_Dense600_' + str(i) + 'nd_err' + str(err*100)
            model.save_weights(foldername+filename+'.hdf5')
        except:
            os.mkdir('./task1_trainall/')
            filename='task1_trainall_Dense600_' + str(i) + 'nd_err' + str(err*100)
            model.save_weights(foldername+filename+'.hdf5')            


    err_ls.append(err*100)
    tst_EER1_ls.append(tst_EER1*100)
    tst_EER2_ls.append(tst_EER2*100)
    
    X_train = None
    X_test = None
    Anchor = None
    Positive = None
    Negative = None
    Anchor_test = None
    Positive_test = None
    Negative_test = None
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('epochs')
ax.set_ylabel('loss')
ax.plot(total_loss_history)
ax.plot(total_valloss_history)
ax.legend(['loss','val_loss'])
fig.savefig('Task_trainall_training_loss')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('Number of Training')
ax2.set_ylabel('accuracy')
ax2.plot(err_ls)
ax2.plot(tst_EER1_ls)
ax2.plot(tst_EER2_ls)
ax2.legend(['train (on dev)', 'train (on test)', 'train+dev (on test)'])
fig2.savefig('Task1_trainall_acc')

# history = []
# err_ls = []
# for iteration in range(1,20000):
    # print(20*'-'+'epochs = {}'.format(iteration)+'-'*20)
    # result = model.fit(trn_ivector_img,trn_trials_onehot,batch_size=1024,epochs=1,validation_data=(dev_ivector_img, dev_trails_onehot))
    # history.append(result.history['loss'])
    # score = model.predict(dev_ivector.reshape(-1,20,30,1))
    # err = calculate_EER(dev_trials, score[:,1])
    # err_ls.append(err)
    # if err < 0.02:
        # model.save_weights('./task1/weight_Tri600_for_err_'+str(err*100)[:4]+'.hdf5')