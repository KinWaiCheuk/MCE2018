import numpy as np
import dataloader

def report_task1_EER(predictions, labels, bl_ivector, spk_mean):
    # This accuracy calcultion is provided by MCE2018
    scores = spk_mean.dot(predictions.transpose())
    blscores = spk_mean.dot(bl_ivector.transpose()) # This will be used in normalization and task 2
    
    # Multi-target normalization
    mnorm_mu = np.mean(blscores,axis=1) 
    mnorm_std = np.std(blscores,axis=1)
    for iter in range(np.shape(scores)[1]):
        scores[:,iter]= (scores[:,iter] - mnorm_mu) / mnorm_std
    pred_scores = np.max(scores,axis=0)

    # Top-S detector EER
    print('\nDev set score using train set :')
    err = dataloader.calculate_EER(labels, pred_scores)
    return err