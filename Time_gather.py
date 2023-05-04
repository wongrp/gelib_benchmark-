import cnine 
from gelib import * 
from datetime import datetime 
from GElib_Operations import * 
import matplotlib.pyplot as plt

num_trials = 50
N_range = [100,200]

plt.figure()
for N in N_range: 
    gelib_ops = GElib_Operations(1,[N,1],5,1)
    sparsities = np.arange(0,1,0.01)
    # time gather
    gather_times_gelib = np.zeros((len(sparsities),num_trials))
    einsum_times_gelib = np.zeros((len(sparsities),num_trials))

    for trial in range(num_trials):
        for id, i in enumerate(sparsities):
            _ , dt = gelib_ops.get_gather(i)
            gather_times_gelib[id,trial] = dt 
            _ , dt = gelib_ops.get_einsum_gather(i) 
            einsum_times_gelib[id,trial] = dt


    gather_times_gelib = np.mean(gather_times_gelib, axis = -1)
    einsum_times_gelib = np.mean(einsum_times_gelib, axis = -1)

    plt.plot(sparsities,gather_times_gelib, label = "gather N={}".format(N))
    plt.plot(sparsities,einsum_times_gelib, '--',label = "einsum N={}".format(N))
plt.xlabel("Sparsity")
plt.ylabel("Time Elapsed (s)")
plt.legend()
plt.savefig("gather_times")
# print(gather_times_gelib)
# ir = e3nn_ops.ir
# F = e3nn_ops.ir_rand
# G = e3nn_ops.get_CGproduct(1)
# R = o3.rand_matrix()
# D = ir.D_from_matrix(R)
# plt.figure()
# plt.imshow(D, cmap='bwr', vmin=-1, vmax=1);
# plt.savefig('D')

