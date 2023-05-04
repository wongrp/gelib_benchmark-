from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from E3nn_Operations import * 
from GElib_Operations import * 
import gc
import os

e3nn_ops = E3nn_Operations(100,0,128,cgmode = "part")
ti = datetime.now()
cg = e3nn_ops.get_CGproduct()
tf = datetime.now()
dt = (tf-ti).total_seconds()
print("e3nn cg product took {}s".format(dt))
ti = datetime.now()
cg = e3nn_ops.get_CGproduct()
tf = datetime.now()
dt = (tf-ti).total_seconds()
print("e3nn cg product took {}s".format(dt))
print("output size {}".format(cg.size()))
del e3nn_ops
gc.collect()
torch.cuda.empty_cache()
e3nn_ops = E3nn_Operations(100,10,128, cgmode = "part")
ti = datetime.now()
cg = e3nn_ops.get_CGproduct()
torch.cuda.synchronize()
tf = datetime.now()
dt = (tf-ti).total_seconds()
print("e3nn cg product took {}s".format(dt))
ti = datetime.now()
cg = e3nn_ops.get_CGproduct()
torch.cuda.synchronize()
tf = datetime.now()
dt = (tf-ti).total_seconds()
print("e3nn cg product took {}s".format(dt))
ti = datetime.now()
cg = e3nn_ops.get_CGproduct()
torch.cuda.synchronize()
tf = datetime.now()
dt = (tf-ti).total_seconds()
print("e3nn cg product took {}s".format(dt))
ti = datetime.now()
cg = e3nn_ops.get_CGproduct()
cg0 = cg[0,0].item()
tf = datetime.now()
dt = (tf-ti).total_seconds()
print("e3nn cg product took {}s".format(dt))



del e3nn_ops 
gc.collect()
torch.cuda.empty_cache()

gelib_ops = GElib_Operations(1,[100,1],10,128)

ti = datetime.now()
cg = gelib_ops.get_CGproduct()
tf = datetime.now()
dt = (tf-ti).total_seconds()
print("gelib cg product took {}s".format(dt))

print(cg.size())

ti = datetime.now()
cg = gelib_ops.get_CGproduct()
torch.cuda.synchronize()
tf = datetime.now()
dt = (tf-ti).total_seconds()
print("gelib cg product took {}s".format(dt))

print(cg.size())


# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)
# start_event.record()

# gelib_ops.get_CGproduct()
# torch.cuda.synchronize()  # Wait for the events to be recorded!
# end_event.record()
# elapsed_time_ms = start_event.elapsed_time(end_event)
# print(elapsed_time_ms)