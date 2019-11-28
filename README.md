# MCE2018
Source code for [Latent space representation for multi-target speaker detection and identification with a sparse dataset using Triplet neural networks](https://arxiv.org/abs/1910.01463). [MCE2018](https://www.kaggle.com/kagglesre/blacklist-speakers-dataset) is used as the dataset for this model

# Dependencies
Tensorflow 1.10.0

keras 2.2.4

sklearn 0.20.3

numpy 1.14.5

matplotlib 3.0.2

# Dataset
1. Download the data from [MCE2018](https://www.kaggle.com/kagglesre/blacklist-speakers-dataset).

2. Unzip everything

3. When loading the i-vectors for the first time, you need to use this code
```python
# Loading i-vector
trn_bl_id, trn_bl_utt, trn_bl_ivector = dataloader.load_ivector('data/trn_blacklist.csv')
trn_bg_id, trn_bg_utt, trn_bg_ivector = dataloader.load_ivector('data/trn_background.csv')
dev_bl_id, dev_bl_utt, dev_bl_ivector = dataloader.load_ivector('data/dev_blacklist.csv')
dev_bg_id, dev_bg_utt, dev_bg_ivector = dataloader.load_ivector('data/dev_background.csv')
tst_id, test_utt, tst_ivector = dataloader.load_ivector('data/tst_evaluation.csv')
```

If you want to load the data faster next time, you can pickle all of them.
The example codes in this repository assume that you pickle all data, and load them back using pickle 
```python
# Getting i-vectors
trn_bl_ivector, trn_bg_ivector, dev_bl_ivector, dev_bg_ivector, tst_ivector = dataloader.get_ivectors()

# Loading labels for task 2
trn_bl_id, trn_bg_id, dev_bl_id, dev_bg_id, tst_id = dataloader.get_spk_ids()
```

Just comment out this chunk of code in cell 4 and replace it with the `dataloader.load_ivector` method
