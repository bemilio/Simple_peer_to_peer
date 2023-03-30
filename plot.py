import matplotlib as mpl
mpl.interactive(True)
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib import gridspec

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "STIXGeneral",
    "font.serif": ["Computer Modern Roman"],
})
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import networkx as nx
import numpy as np
import pickle
import torch
import os
from itertools import cycle

TICH = 0
HSDM = 1
FBF=2

method_name= { TICH: "Tich", HSDM:"HSDM", FBF:"FBF"}
load_files_from_current_dir = True
if load_files_from_current_dir:
    directory = "."
else:
    directory = r"/..."
if not os.path.exists(directory + "/Figures"):
    os.makedirs(directory + r"/Figures")

f = open('saved_test_result_0.pkl', 'rb')
[x_store, residual_store, dual_share_store, dual_loc_store,
local_constr_viol, shared_const_viol,
loc_const_viol_tvar, shared_const_viol_tvar,
distance_from_optimal_tvar, edge_to_index, N_iter_per_timestep ] = pickle.load(f)


f.close()

N_tests = distance_from_optimal_tvar.size(0)
T = distance_from_optimal_tvar.size(2)
Steps_between_iterations = 10
N_iterations = residual_store.size(1)


torch.Tensor.ndim = property(lambda self: len(self.shape))  # Necessary to use matplotlib with tensors

#### Plot 1: residual static case
fig, ax = plt.subplots(1, 1, figsize=(4, 2.1), sharex='col')
x =  range(1, Steps_between_iterations * N_iterations, Steps_between_iterations)
ax.plot(x, torch.mean(residual_store, dim=0))
ax.grid(True)
ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_ylabel(r'$V({\omega}^k)$', fontsize=9)
ax.set_xlabel("Iteration", fontsize=9)
ax.set_xlim(1, N_iterations*Steps_between_iterations)

plt.tight_layout()
fig.tight_layout()
plt.draw()

plt.show(block=False)

fig.savefig(directory + '/Figures/Residual.png')
fig.savefig(directory + '/Figures/Residual.pdf')

#### Plot 2: Distance between time varying and static solution
fig, ax = plt.subplots(1, 1, figsize=(4, 2.1), sharex='col')
x =  range(T)
lines = ["-", "--", "-.", ":"]
linecycler = cycle(lines)
for index_K in range(len(N_iter_per_timestep)):
    ax.plot(x, torch.mean(distance_from_optimal_tvar[:,index_K,:], dim=0), label=str(N_iter_per_timestep[index_K]),linestyle=next(linecycler))
ax.grid(True)
ax.set_ylabel(r'$\|x_t - x_t^*\|$', fontsize=9)
ax.set_xlabel("Time-step", fontsize=9)
ax.set_yscale('log')
# ax[0,0].set_xlim(1, T)
ax.legend(title=r'$K$', fontsize=7)

plt.tight_layout()
fig.tight_layout()
plt.draw()

plt.show(block=False)

fig.savefig(directory + '/Figures/Distance_tvar.png')
fig.savefig(directory + '/Figures/Distance_tvar.pdf')

#### Plot 3: Constraint violation tvar
fig, ax = plt.subplots(1, 1, figsize=(4, 2.1), sharex='col')
x =  range(T)
lines = ["-", "--", "-.", ":"]
linecycler = cycle(lines)
for index_K in range(len(N_iter_per_timestep)):
    ax.plot(x, torch.mean(shared_const_viol_tvar[:,index_K,:]+loc_const_viol_tvar[:,index_K,:], dim=0),\
        label=str(N_iter_per_timestep[index_K]),linestyle=next(linecycler))
ax.grid(True)
ax.set_ylabel(r'$\|Ax_t - b_t\|$', fontsize=9)
ax.set_xlabel("Time-step", fontsize=9)
ax.set_yscale('log')
# ax[0,0].set_xlim(1, T)

plt.tight_layout()
fig.tight_layout()
plt.draw()
ax.legend(title=r'$K$', fontsize=7)
plt.show(block=False)

fig.savefig(directory + '/Figures/Constr_viol.png')
fig.savefig(directory + '/Figures/Constr_viol.pdf')



print("Done")