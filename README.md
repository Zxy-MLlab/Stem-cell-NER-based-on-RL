# Stem-cell-NER-based-on-RL
# Introduce
This code is the core part of Lader model. Lader is composed of a labeler based on the BioBERT pre-trained model without fine-tuning and a distiller based on reinforcement learning, which realizes the high-precision labeling of stem cells without manual supervision.
# Dataset
All data are from PubMed database.\
\
The dataset contains the following two files：
* **train_sentence_label.json:** This is a completely unsupervised rule matching labeled stem cell label dataset, including 18849 sentences and 22614 noisy stem cell labels, which is used to train the model.
* **precise_sentence_label.json:** This is our manually labeled stem cell label dataset, including 1479 sentences and 1577 stem cell labels, which is used for the evaluation of the model.

**Pre-trained language model**\
We use the biobert model based on the pytorch version. You need to download it and put it in biobert_base_cased/.

**Stem_cell_NER_dataset**\
This dataset stores the stem cell named entity recognition dataset provided by us, including 4345 sentences and 4491 stem cell labels (accuracy: more than 90%).
# Requirements
* Python (>=3.6.1)
* TensorFlow (=1.6.0)
* torch (=1.0.1)
# Run
We have completed the first stage of labeling and training data construction and put it in the rlmodel/data, so you can train RL model by executing the following code.
```
python rlmodel.py
```  
# Evaluation
You can perform the following procedure to evaluate the model on a manually labeled stem cell dataset.
```
python do_devel.py
``` 
# Reference
[Feng et al. 2016] Jun Feng, Minlie Huang, Li Zhao, Yang Yang, and Xiaoyan Zhu. Reinforcement Learning for Relation Classification from Noisy Data. In AAAI2018.\
(https://github.com/xuyanfu/TensorFlow_RLRE)
