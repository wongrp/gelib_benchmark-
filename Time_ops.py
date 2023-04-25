from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from E3nn_Operations import * 
from GElib_Operations import * 

# irrep parameters 
num_irreps = 100
num_trials = 10
max_l_max = 6
num_channels_min = 1
num_channels_max = 5
num_channels_increment = 1

# parameter arrays 
max_l_range = range(max_l_max+1)
num_channels_range = np.arange(num_channels_min,num_channels_max+1,num_channels_increment)
channel_id_max = int((num_channels_max-num_channels_min)/num_channels_increment+1)

# cg product time arrays
CG_times_e3nn = np.zeros((max_l_max+1,channel_id_max,num_trials))
CG_times_gelib = np.zeros((max_l_max+1,channel_id_max,num_trials))

# dummy operation and discard
dummy_e3nn_ops = E3nn_Operations(1, 0,1)
dummy_e3nn_ops.get_CGproduct()

# time the cg product 
channel_id = -1 
for max_l in max_l_range:
    channel_id = -1
    for num_channels in num_channels_range:
        channel_id += 1 
        for trial_id in range(num_trials):
            print("trial max_l={},num_channels={},trial_id={}".format(max_l,num_channels,trial_id))
            e3nn_ops = E3nn_Operations(num_irreps, max_l,int(num_channels))
            gelib_ops = GElib_Operations(num_irreps,max_l,num_channels)

            ti = datetime.now() 
            e3nn_ops.get_CGproduct()
            tf = datetime.now() 
            dt = (tf-ti).total_seconds() 
            CG_times_e3nn[max_l,channel_id,trial_id]= dt
            
            ti = datetime.now() 
            gelib_ops.get_CGproduct()
            tf = datetime.now() 
            dt = (tf-ti).total_seconds() 
            CG_times_gelib[max_l,channel_id,trial_id]= dt

CG_times_e3nn = np.mean(CG_times_e3nn,axis = 2)
CG_times_gelib = np.mean(CG_times_gelib,axis = 2)

plt.figure()
for i in range(channel_id_max):
    plt.plot(max_l_range,CG_times_e3nn[:,i],'--',label = "{} channels; E3nn".format(int(num_channels_max/channel_id_max*i+num_channels_min)))
    plt.plot(max_l_range,CG_times_gelib[:,i],label = "{} channels; GElib".format(int(num_channels_max/channel_id_max*i+num_channels_min)))
plt.xlabel("Maximum l")
plt.ylabel("Time Elapsed (s)")
plt.legend()
plt.savefig("plot")

# time gather
# gather_times_gelib = []
# for i in np.arange(0,1,0.05):
#     ti = datetime.now()
#     gelib_ops.get_gather(i)
#     tf = datetime.now()
#     dt = (tf-ti).total_seconds()   
#     gather_times_gelib.append(dt) 

# print(gather_times_gelib)
# ir = e3nn_ops.ir
# F = e3nn_ops.ir_rand
# G = e3nn_ops.get_CGproduct(1)
# R = o3.rand_matrix()
# D = ir.D_from_matrix(R)
# plt.figure()
# plt.imshow(D, cmap='bwr', vmin=-1, vmax=1);
# plt.savefig('D')


