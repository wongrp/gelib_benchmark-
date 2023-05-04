from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import lines
import pandas as pd

from E3nn_Operations import * 
from GElib_Operations import * 
import gc
import os

# folder
folder = "Time_CG_May3/SO3vecArr"
if not os.path.exists(folder):
    os.makedirs(folder)

# test part of vec? 
cgmode = "vec"
# with e3nn? 
with_mflops = False
with_e3nn = True
with_gelib = True
proportional_batch_size = False
proportional_num_channels = False

# irrep parameters 
batch_size = 1
num_irreps_range = [25]
num_trials = 10
max_l_min = 0
max_l_max = 5

# parameter arrays 
max_l_range = np.arange(max_l_min,max_l_max+1,1)
num_channels_range = [128,196]
channel_id_max = len(num_channels_range)

# cg product time arrays
CG_times_e3nn = np.zeros((len(num_irreps_range),max_l_max+1,channel_id_max,num_trials))
CG_times_gelib = np.zeros((len(num_irreps_range),max_l_max+1,channel_id_max,num_trials))

# dummy operation and discard
# dummy_e3nn_ops = E3nn_Operations(1,0,1)
# dummy_e3nn_ops.get_CGproduct()
# dummy_gelib_ops = GElib_Operations(1,[1,1],0,1)
# dummy_gelib_ops.get_CGproduct() 

# batch size 
b_times_N = num_irreps_range[-1]
c_times_N = num_irreps_range[-1]
if proportional_num_channels == True: 
    num_channels_range =  []
    for num_irreps in num_irreps_range: 
        num_channels = int(c_times_N/num_irreps)
        num_channels_range.append(num_channels)
print(num_channels_range)
batch_size_range = []


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
        for num_channels in num_channels_range:
            channel_id += 1 
            for trial_id in range(num_trials):
                print("trial max_l={},num_channels={},trial_id={}".format(max_l,num_channels,trial_id))
                e3nn_ops = E3nn_Operations(num_irreps, max_l,num_channels,cgmode =cgmode)
                gelib_ops = GElib_Operations(batch_size,[num_irreps,1],max_l,num_channels,cgmode =cgmode)
                
                if with_e3nn == True:
                    # let's try a dummy operation here
                    e3nn_ops.get_CGproduct() 
                    ti = datetime.now() 
                    e3nn_ops.get_CGproduct()
                    torch.cuda.synchronize()
                    tf = datetime.now() 
                    dt = (tf-ti).total_seconds() 
                    CG_times_e3nn[num_irreps_id,max_l,channel_id,trial_id]= dt
                if with_gelib == True: 
                    # let's try a dummy operation here
                    gelib_ops.get_CGproduct()
                    ti = datetime.now() 
                    gelib_ops.get_CGproduct()
                    torch.cuda.synchronize()
                    tf = datetime.now() 
                    dt = (tf-ti).total_seconds() 
                    CG_times_gelib[num_irreps_id,max_l,channel_id,trial_id]= dt

                # clear memory
                del e3nn_ops.ir_rand
                del gelib_ops.F
                del e3nn_ops
                del gelib_ops
                gc.collect()
                torch.cuda.empty_cache()
                


if with_mflops == True and cgmode == "part":
    # read mflops from log file 
    mflops_arr =np.loadtxt(open('CGproducts.csv'),delimiter = ',',usecols = 8)
    # Remove every other operation (dummy ops) starting with the first 
    mflops_arr = np.delete(mflops_arr, np.arange(0,mflops_arr.size,2))
    mflops_arr = mflops_arr.flatten()
    mflops_arr = mflops_arr.reshape((len(num_irreps_range),max_l_max+1,channel_id_max,num_trials)) # or reshape? 
    CG_mflops_gelib = np.mean(mflops_arr,axis = -1)

CG_times_e3nn = np.mean(CG_times_e3nn,axis = -1)
CG_times_gelib = np.mean(CG_times_gelib,axis = -1)


f1 = plt.figure(1)
ax1 = f1.add_subplot()
f2 = plt.figure(2)
ax2 = f2.add_subplot()
f3 = plt.figure(3)
ax3 = f3.add_subplot()

# Plot empty lines for legend 
handle_list = []
if with_e3nn == True and with_gelib == True: 
    handle_list.append(lines.Line2D([],[],ls='--',c="black", label = "e3nn"))
    handle_list.append(lines.Line2D([],[],c="black",label = "GElib"))

# Plot time data 
for j in range(len(num_irreps_range)):
    for i in range(channel_id_max):
        color=next(ax1._get_lines.prop_cycler)['color'] # FIX ax doesn't exist 
        label = "N={},{} channels".format(num_irreps_range[j],num_channels_range[i])
        handle_list.append(mpatches.Patch(color=color, label=label))
        if with_e3nn == True: 
            ax1.plot(max_l_range,CG_times_e3nn[j,:,i],'--', color = color)
        if with_gelib == True: 
            ax1.plot(max_l_range,CG_times_gelib[j,:,i],color = color) 
            #plot mflops 
            if with_mflops == True and cgmode == 'part': 
                ax3.plot(max_l_range,CG_mflops_gelib[j,:,i],color = color)
        if with_e3nn == True and with_gelib == True: 
            plt.figure(2)
            plt.plot(max_l_range, CG_times_gelib[j,:,i]/CG_times_e3nn[j,:,i],color = color)
        

# labels
ax1.set_xlabel("Maximum L")
ax1.set_ylabel("Time Elapsed (s)")
ax1.legend(handles = handle_list)
f1.savefig("{}/cg_times_{}_N={}_num_channels={}_{}trials_e3nn={}_gelib={}".format(folder,cgmode,num_irreps_range,num_channels_range,num_trials,with_e3nn,with_gelib))

if with_e3nn == True and with_gelib == True: 
    ax2.set_xlabel("Maximum L")
    ax2.set_ylabel("Ratio")
    ax2.legend(handles = handle_list[2:])
    f2.savefig("{}/cg_times_{}_ratio_N={}_num_channels={}_{}trials_e3nn={}_gelib={}".format(folder,cgmode,num_irreps_range,num_channels_range,num_trials,with_e3nn,with_gelib))

if with_mflops == True and cgmode == 'part': 
    plt.figure(3)
    plt.xlabel("Maximum L")
    plt.ylabel("mflops")
    plt.legend(handles = handle_list)
    plt.savefig("{}/cg_mflops_{}_N={}_num_channels={}_{}trials_e3nn={}_gelib={}".format(folder,cgmode,num_irreps_range,num_channels_range,num_trials,with_e3nn,with_gelib))


