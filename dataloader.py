import numpy as np
import pickle
from sklearn.metrics import roc_curve

def creating_id_mapping():
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
        
    return dev2train, test2train

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

def get_ivectors():
    # Loading i-vectors for train set, dev set and test set
    trn_bl_ivector = pickle.load(open('./data/trn_bl_ivector','rb'))
    trn_bg_ivector = pickle.load(open('./data/trn_bg_ivector','rb'))
    dev_bl_ivector = pickle.load(open('./data/dev_bl_ivector','rb'))
    dev_bg_ivector = pickle.load(open('./data/dev_bg_ivector','rb'))
    tst_ivector = pickle.load(open('./data/tst_ivector','rb'))
    
    #length normalization
    trn_bl_ivector = length_norm(trn_bl_ivector)
    trn_bg_ivector = length_norm(trn_bg_ivector)
    dev_bl_ivector = length_norm(dev_bl_ivector)
    dev_bg_ivector = length_norm(dev_bg_ivector)
    tst_ivector = length_norm(tst_ivector)
    
    return trn_bl_ivector, trn_bg_ivector, dev_bl_ivector, dev_bg_ivector, tst_ivector

def get_spk_ids():
    # Loading speaker ID, for task 2
    trn_bl_id = pickle.load(open('./data/trn_bl_id','rb'))
    trn_bg_id = pickle.load(open('./data/trn_bg_id','rb'))
    dev_bl_id = pickle.load(open('./data/dev_bl_id','rb'))
    dev_bg_id = pickle.load(open('./data/dev_bg_id','rb'))
    tst_id = pickle.load(open('./data/tst_id','rb'))
    
    return trn_bl_id, trn_bg_id, dev_bl_id, dev_bg_id, tst_id

def get_spk_utt():
    # Loading speaker utt
    trn_bl_utt = pickle.load(open('./data/trn_bl_utt','rb'))
    trn_bg_utt = pickle.load(open('./data/trn_bg_utt','rb'))
    dev_bl_utt = pickle.load(open('./data/dev_bl_utt','rb'))
    dev_bg_utt = pickle.load(open('./data/dev_bg_utt','rb'))
    test_utt = pickle.load(open('./data/test_utt','rb'))

    return trn_bl_utt, trn_bg_utt, dev_bl_utt, dev_bg_utt, test_utt

def get_tst_trials():
    # creating test labels, for task 1
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

    return tst_trials, tst_trials_label