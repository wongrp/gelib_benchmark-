from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from E3nn_Operations import * 
from GElib_Operations import * 

# with e3nn? 
with_e3nn = False
proportional_batch_size = False

# irrep parameters 
batch_size = 1  
num_irreps_range = [1,10,100,1000,10000]
num_trials = 10
max_l_max = 6
num_channels_min = 1
num_channels_max = 1
num_channels_increment = 2

# parameter arrays 
max_l_range = range(max_l_max+1) 
if num_channels_max == num_channels_min:
    num_channels_range = [num_channels_max]
else:
    num_channels_range = np.arange(num_channels_min,num_channels_max+1,num_channels_increment)
channel_id_max = int((num_channels_max-num_channels_min)/num_channels_increment+1)

# cg product time arrays
CG_times_e3nn = np.zeros((len(num_irreps_range),max_l_max+1,channel_id_max,num_trials))
CG_times_gelib = np.zeros((len(num_irreps_range),max_l_max+1,channel_id_max,num_trials))

# dummy operation and discard
dummy_e3nn_ops = E3nn_Operations(1, 0,1)
dummy_e3nn_ops.get_CGproduct()
dummy_gelib_ops = GElib_Operations(1,[1,1],0,1)
dummy_gelib_ops.get_CGproduct() 

# batch size 
batch_size_range = []
b_times_N = num_irreps_range[-1]

# time the cg product 
channel_id = -1 
num_irreps_id = -1
for num_irreps in num_irreps_range: 
    if proportional_batch_size == True:
        batch_size = int(b_times_N/num_irreps) 
        batch_size_range.append(batch_size)
    num_irreps_id += 1 
    for max_l in max_l_range:
        channel_id = -1
        for num_channels in num_channels_ran/ls={},trial_id={}".format(max_l,num_channels,trial_id))
                e3nn_ops = E3nn_Operations(num_irreps, max_l,int(num_channels))
                gelib_ops = GElib_Operations(batch_size,[num_irreps,1],max_l,num_channels)

                ti = datetime.now() 
                e3nn_ops.get_CGproduct()
                tf = datetime.now() 
                dt = (tf-ti).total_seconds() 
                CG_times_e3nn[num_irreps_id,max_l,channel_id,trial_id]= dt
                
                ti = datetime.now() 
                gelib_ops.get_DiagCGproduct()
                tf = datetime.now() 
                dt = (tf-ti).total_seconds() 
                CG_times_gelib[num_irreps_id,max_l,channel_id,trial_id]= dt

CG_times_e3nn = np.mean(CG_times_e3nn,axis = -1)
CG_times_gelib = np.mean(CG_times_gelib,axis = -1)

plt.figure()
for j in range(len(num_irreps_range)):
    for i in range(channel_id_max):
        if with_e3nn == True: 
            plt.plot(max_l_range,CG_times_e3nn[j,:,i],'--',label = "N={},{} channels; E3nn".format(
                num_irreps_range[j],int(num_channels_max/channel_id_max*i+num_channels_min)))
        plt.plot(max_l_range,CG_times_gelib[j,:,i],label = "b={},N={},{} channels; GElib".format(batch_size_range[j],
            num_irreps_range[j],int(num_channels_max/channel_id_max*i+num_channels_min)))
plt.xlabel("Maximum l")
plt.ylabel("Time Elapsed (s)")
plt.legend()
plt.savefig("diagcg_times_N={}_chanmin={}_chanmax={}_e3nn={}".format(num_irreps_range,num_channels_min,num_channels_max,with_e3nn))



