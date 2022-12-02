import numpy as np

import torch as th
import torch.nn as nn


def contrastive_loss(y1, y2, flag):
    '''
    flag indicates whether y1,y2 belong to the same class
    flag = 0 (same class)
    flag = 1 (different class)
    '''
    

class SiameseNetwork(nn.Module):
    '''
    Siamese network to generate embeddings for our labels
    '''
    def __init__(self):
        super(SiameseNetwork, self).__init__()

