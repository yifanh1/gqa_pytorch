# gqa_pytorch
Basic model for visual question answering  
Data: https://cs.stanford.edu/people/dorarad/gqa/index.html  
  Major part of preprocess.py and dataset.py is from https://github.com/ronilp/mac-network-pytorch-gqa  
 Attention model for VQA: https://arxiv.org/abs/1707.07998
  Simplify some functions to accelerate the process.
  
  ### notes:
  - train_all.py is training file for basic CNN+LSTM model  
  - train_att.py is training file for attention model  
  - train_double_att.py is training file for double attention model  
  - model.py is the model file for basic model  
  - attention.py is the model file for attention model  
  - attention2.py is the model file for double attention model  
## TODO:  
1. TEST.py  (waiting for the official test set releasing)
2. Attention.  (done)
3. Add a channel for Graph.
4. Try the idea of Bi-Directional Attention Flow. (done)
