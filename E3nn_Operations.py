import torch_geometric
from torch_geometric.data import Data
from e3nn import o3 
from copy import copy 
import matplotlib.pyplot as plt
from datetime import datetime 
import torch

class E3nn_Operations():
    def __init__(self,num_irreps, max_l, num_channels, cgmode = "part"):
        self.max_l = int(max_l)
        self.num_irreps = num_irreps  
        self.parity = 1 
        self.num_channels = num_channels          
        
        
        # output a part of vec? 
        if cgmode == "vec":
            irreps_list = []
            for l in range(max_l+1): 
                irreps_list.append((num_channels,(l,self.parity)))
                print(irreps_list)
            self.ir = o3.Irreps(irreps_list)
            self.ir_rand = self.ir.randn(num_irreps,-1,device = torch.device('cuda'))
            self.tp = o3.FullTensorProduct(self.ir,self.ir)
            
        elif cgmode == "part":
            irrep_str = str(num_channels)+"x"+str(self.max_l)+"e"
            irrep_str_out = str(num_channels**2)+"x"+str(self.max_l)+"e"
            self.ir = o3.Irreps(irrep_str)  
            self.ir_out = o3.Irreps(irrep_str_out)
            self.ir_rand = self.ir.randn(num_irreps,-1,device = torch.device('cuda'))
            self.tp = o3.FullyConnectedTensorProduct(irreps_in1 = self.ir, irreps_in2 = self.ir,irreps_out=self.ir_out)
        
        print(self.ir)
        self.tp.cuda() #need otherwise device mismatch
        self.dtp = tp = o3.ElementwiseTensorProduct(self.ir,self.ir)

        # def get_gather(self,sparsity):
        #     # initialize mask
        #     C = torch.empty(N,N)
        #     torch.nn.init.sparse_(C,sparsity = sparsity)
        
        #     # gather
        #     G = self.F.gather(Cmask) 
        #     return G

    def get_CGproduct(self): 
        return self.tp(self.ir_rand, self.ir_rand)


    def get_DiagCGproduct(self): 
        return self.tp(self.ir_rand,self.ir_rand)


