from keras.layers import Input, Conv2D, Lambda, concatenate, Dense, Flatten,MaxPooling2D,Activation
from keras.models import Model, Sequential
from keras import backend as K
import numpy as np
import random
from itertools import combinations

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
            A_P_pairs = random.sample(list(combinations(same_class_idx,2)),k=ap_pairs) #Generating Anchor-Positive pairs
        else:
            same_class_sampleer_idx = random.choice(range(len(same_class_idx)-21))
            A_P_pairs = random.sample(list(combinations(same_class_idx[same_class_sampleer_idx:same_class_sampleer_idx+21],2)),k=ap_pairs) #Generating Anchor-Positive pairs
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

def triplet_loss(y_true, y_pred, alpha = 0.5):
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

def create_TNN():
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
    model.summary()
    
    # after training the model, we only need anchor_input and its output encoded_anchor
    return model, anchor_input, encoded_anchor
