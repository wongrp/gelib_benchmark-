import cnine 
from gelib import * 
import torch
import numpy as np
from copy import copy
from datetime import datetime 

class GElib_Operations: 
    def __init__(self,batch_size,adim, max_l, num_channels,cgmode = "part"): 
        self.b = batch_size
        self.adim = adim 
        self.N = adim[0]
        self.max_l = max_l
        self.num_channels = num_channels 
        self.tau = num_channels*np.ones((max_l+1)).astype(int)

        if cgmode == "vec": 
            self.F = SO3vecArr.randn(self.b, adim, self.tau, device = "cuda")
            self.args = [self.F,self.F] 
        elif cgmode == "part":
            self.F = SO3partArr.randn(self.b,adim,max_l,num_channels, device = "cuda")
            self.args = [self.F,self.F,max_l]
    
    def get_gather(self,sparsity):
        # initialize mask
        C = torch.empty(self.N,self.N)
        torch.nn.init.sparse_(C,sparsity = sparsity)
        Cmask = cnine.Rmask1(C)
        
        # gather
        ti = datetime.now() 
        G = self.F.gather(Cmask)
        tf = datetime.now()
        dt = (tf-ti).total_seconds() 
        return G,dt

    def get_einsum_gather(self,sparsity): 
        # initialize mask
        C = torch.empty(self.N,self.N)
        torch.nn.init.sparse_(C,sparsity = sparsity)
        G = SO3vecArr.zeros(self.b, self.adim, self.tau)
        C = C.to(torch.complex64)
        # gather
        ti = datetime.now() 
        for l in range(self.max_l):
            G.parts[l] = torch.einsum("ik,bkjnc-> bijnc",C,self.F.parts[l])
        tf = datetime.now()

        dt = (tf-ti).total_seconds() 
        return G,dt

    def get_CGproduct(self): 
        return CGproduct(*self.args)


    def get_DiagCGproduct(self): 
        return DiagCGproduct(self.F, self.F)



