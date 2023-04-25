import cnine 
from gelib import * 
import torch
import numpy as np
from copy import copy

class GElib_Operations: 
    def __init__(self,N, max_l, num_channels): 
        self.N = N 
        self.max_l = max_l
        self.num_channels = num_channels 
        self.tau = num_channels*np.ones((max_l+1)).astype(int)

        self.F = SO3vecArr.randn(1, [N,1], self.tau) 

    def get_gather(self,sparsity):
        # initialize mask
        C = torch.empty(self.N,self.N)
        torch.nn.init.sparse_(C,sparsity = sparsity)
        Cmask = cnine.Rmask1(C)
        
        # gather
        G = self.F.gather(Cmask) 
        return G

    def get_CGproduct(self): 
        return CGproduct(self.F,self.F)

    def get_DiagCGproduct(self): 
        return DiagCGproduct(self.F, self.F)



