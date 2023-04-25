from torch_geometric.data import Data
from e3nn import o3 
from copy import copy 
import matplotlib.pyplot as plt
from datetime import datetime 

class E3nn_Operations:
    def __init__(self,num_irreps, max_l, num_channels):
        self.max_l = max_l
        self.num_irreps = num_irreps  
        self.parity = 1 
        self.num_channels = num_channels 
 
        # construct irreps
        irreps_list = []
        for l in range(max_l+1): 
            irreps_list.append((num_channels,(l,self.parity)))
        print(irreps_list)
        self.ir = o3.Irreps(irreps_list)
        self.ir_rand = self.ir.randn(num_irreps,-1)
        self.tp = o3.FullTensorProduct(self.ir,self.ir)
        self.dtp = tp = o3.ElementwiseTensorProduct(self.ir,self.ir)

        # def get_gather(self,sparsity):
        #     # initialize mask
        #     C = torch.empty(N,N)
        #     torch.nn.init.sparse_(C,sparsity = sparsity)
        
        #     # gather
        #     G = self.F.gather(Cmask) 
        #     return G

    def get_CGproduct(self): 
        return self.tp(self.ir_rand,self.ir_rand)


    def get_DiagCGproduct(self): 
        return self.tp(self.ir_rand,self.ir_rand)


