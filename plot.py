import matplotlib as mpl
import seaborn as sns
import pandas as pd
mpl.interactive(True)
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib import gridspec

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "STIXGeneral",
    "font.serif": ["Computer Modern Roman"],
})
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
load_files_from_current_dir = False
create_new_dataframe_file = True #TODO: Remove, deprecated
if load_files_from_current_dir:
    directory = "."
else:
    directory = r"/Users/ebenenati/surfdrive/TUDelft/Simulations/Cross_interference/Tichonov_vs_HSDm/23_03_23"
if not os.path.exists(directory + "/Figures"):
    os.makedirs(directory + r"/Figures")

if create_new_dataframe_file or not os.path.exists('saved_dataframe.pkl'):
    if  create_new_dataframe_file:
        print("DataFrame file will be recreated as requested. This will take a while.")
    else:
        print("File containing the DataFrame not found, it will be created. This will take a while.")
    #### Toggle between loading saved file in this directory or all files in a specific directory
    if load_files_from_current_dir:
        f = open('saved_test_result.pkl', 'rb')
        x_store_tich, x_store_hsdm, x_store_std, \
            dual_store_tich, dual_store_hsdm, dual_store_std,  \
            aux_store_tich, aux_store_hsdm, aux_store_std, \
            residual_store_tich, residual_store_hsdm, residual_store_std, \
            sel_func_store_tich, sel_func_store_hsdm, sel_func_store_std, \
            parameters_tested= pickle.load(f)
        f.close()
    else:
        #########
        # Load all files in a directory and stack them in single tensors
        #########
        N_files = 0
        for filename in os.listdir(directory):
            if filename.find('.pkl')>=0:
                N_files=N_files+1 #count all files

        dummy = False
        for filename in os.listdir(directory):
            if filename.find('.pkl')>=0:
                f=open(directory+"/"+filename, 'rb')
                x_store_tich_single_file, x_store_hsdm_single_file, x_store_std_single_file, \
                    dual_store_tich_single_file, dual_store_hsdm_single_file, dual_store_std_single_file, \
                    aux_store_tich_single_file, aux_store_hsdm_single_file, aux_store_std_single_file, \
                    residual_store_tich_single_file, residual_store_hsdm_single_file, residual_store_std_single_file, \
                    sel_func_store_tich_single_file, sel_func_store_hsdm_single_file, sel_func_store_std_single_file, \
                    parameters_tested_tich, parameters_tested_hsdm = pickle.load(f)
                if not dummy:
                    x_store_tich= x_store_tich_single_file
                    x_store_hsdm = x_store_hsdm_single_file
                    x_store_std = x_store_std_single_file
                    residual_store_tich = residual_store_tich_single_file
                    residual_store_hsdm= residual_store_hsdm_single_file
                    residual_store_std = residual_store_std_single_file
                    sel_func_store_tich = sel_func_store_tich_single_file
                    sel_func_store_hsdm = sel_func_store_hsdm_single_file
                    sel_func_store_std = sel_func_store_std_single_file
                else:
                    x_store_tich = torch.cat((x_store_tich, x_store_tich_single_file),dim=0)
                    x_store_hsdm = torch.cat((x_store_hsdm, x_store_hsdm_single_file),dim=0)
                    x_store_std = torch.cat((x_store_std, x_store_std_single_file),dim=0)
                    residual_store_tich = torch.cat((residual_store_tich, residual_store_tich_single_file),dim=0)
                    residual_store_hsdm = torch.cat((residual_store_hsdm, residual_store_hsdm_single_file),dim=0)
                    residual_store_std = torch.cat((residual_store_std, residual_store_std_single_file),dim=0)
                    sel_func_store_tich = torch.cat((sel_func_store_tich, sel_func_store_tich_single_file),dim=0)
                    sel_func_store_hsdm = torch.cat((sel_func_store_hsdm, sel_func_store_hsdm_single_file),dim=0)
                    sel_func_store_std = torch.cat((sel_func_store_std, sel_func_store_std_single_file),dim=0)
                dummy = True
    print("Files loaded...")
    N_tests = x_store_tich.size(0)
    N_tested_sets_of_params_tich = len(parameters_tested_tich)
    N_tested_sets_of_params_hsdm = len(parameters_tested_hsdm)
    N_parameters = 3 # parameters_to_test is a list of tuples, each tuple contains N_parameters
    N_agents = x_store_tich.size(2)
    N_opt_var = x_store_tich.size(3)
    Steps_between_iterations = 10
    N_iterations = residual_store_tich.size(2)
    N_methods = 3 # tichonov, hsdm, standard


    torch.Tensor.ndim = property(lambda self: len(self.shape))  # Necessary to use matplotlib with tensors

    #### Plot 1
    # Plot Tichonov against FBF
    weight_reg_to_plot = .6
    approx_reg_to_plot = 2
    alpha_to_plot = 1.

    hsdm_weight_reg_to_plot = .7

    for i in range(len(parameters_tested_tich)):
        if parameters_tested_tich[i][0] == weight_reg_to_plot and \
                parameters_tested_tich[i][1] == alpha_to_plot and \
                parameters_tested_tich[i][2] == approx_reg_to_plot:
            index_tich_to_plot = i

    for i in range(len(parameters_tested_hsdm)):
        if parameters_tested_hsdm[i][0] == hsdm_weight_reg_to_plot:
            index_hsdm_to_plot = i

    Steps_between_iterations = 10
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    # gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1],
    #                        wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845)
    # fig, ax = plt.subplots(gs)
    fig, ax = plt.subplots(2, 2, figsize=(4, 2.1), sharex='col', width_ratios=[4, 1])
    x_1 = range(1, Steps_between_iterations * N_iterations, Steps_between_iterations)
    x_2 = range(Steps_between_iterations *(N_iterations-9*N_iterations//10), Steps_between_iterations * (N_iterations), Steps_between_iterations)
    # ax[0].plot(x, torch.mean(residual_store_std[:, 0, :], dim=0), color='k', label="FBF")
    # ax[0].fill_between(x, torch.min(residual_store_std[:, 0, :], dim=0)[0].numpy(), \
    #                    y2=torch.max(residual_store_std[:, 0, :], dim=0)[0].numpy(), alpha=0.2, color='k')
    ax[0,0].plot(x_1, torch.mean(residual_store_hsdm[:, index_hsdm_to_plot, :], dim=0), color='g', label="HSDM", linestyle=next(linecycler))
    # ax[0].fill_between(x, torch.min(residual_store_hsdm[:, index_best_hsdm, :], dim=0)[0].numpy(), \
    #                    y2=torch.max(residual_store_hsdm[:, index_best_hsdm, :], dim=0)[0].numpy(), alpha=0.2, color='g')
    ax[0,0].plot(x_1, torch.mean(residual_store_tich[:, index_tich_to_plot, :], dim=0), color='m', label="Alg. 1", linestyle=next(linecycler))
    # ax[0].fill_between(x, torch.min(residual_store_tich[:, index_best_tich, :], dim=0)[0].numpy(), \
    #                    y2=torch.max(residual_store_tich[:, index_best_tich, :], dim=0)[0].numpy(), alpha=0.2, color='m')
    ax[0,0].grid(True, which='both', axis='both')
    ax[0,0].set_yscale('log')
    ax[0,0].set_xscale('log')
    ax[0,0].set_ylabel("Residual", fontsize=9)
    ax[0,0].set_xlim(1, N_iterations* Steps_between_iterations)
    ax[0,0].set_ylim(10 ** (-4), 10 ** (2))
    ax[0,0].set_yticks([10**(-4), 10 ** (-2), 1, 10**(2)])
    ax[0,0].legend(fontsize=9)

    # ax[1].plot(x, torch.mean(sel_func_store_std[:, 0, :], dim=0), color='k', label="FBF")
    # ax[1].fill_between(x, torch.min(sel_func_store_std[:, 0, :], dim=0)[0].numpy(), \
    #                    y2=torch.max(sel_func_store_std[:, 0, :], dim=0)[0].numpy(), alpha=0.2, color='k')
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    relative_advantage_hsdm = 100*(sel_func_store_hsdm[:, index_hsdm_to_plot, :] - sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_hsdm[:, index_hsdm_to_plot, :].size(1)) )\
                              / torch.abs(sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_hsdm[:, index_hsdm_to_plot, :].size(1)))
    ax[1,0].plot(x_1, torch.mean(relative_advantage_hsdm, dim=0), color='g', label="HSDM",linestyle=next(linecycler))
    # ax[1].fill_between(x, torch.min(relative_advantage_hsdm, dim=0)[0].numpy(),\
    #                  y2=torch.max(relative_advantage_hsdm, dim=0)[0].numpy(), alpha=0.2, color='g')
    # ax[1].plot(x, torch.mean(sel_func_store_hsdm[:, index_best_hsdm, :], dim=0), color='g', label="HSDM")
    # ax[1].fill_between(x, torch.min(sel_func_store_hsdm[:, index_best_hsdm, :], dim=0)[0].numpy(), \
    #                    y2=torch.max(sel_func_store_hsdm[:, index_best_hsdm, :], dim=0)[0].numpy(), alpha=0.2, color='g')
    relative_advantage_tich = 100*(sel_func_store_tich[:,index_tich_to_plot,:] - sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, index_tich_to_plot, :].size(1)) )\
                              / torch.abs(sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, index_tich_to_plot, :].size(1)))
    # relative_advantage_tich = sel_func_store_tich[:, index_best_tich, :]
    ax[1,0].plot(x_1, torch.mean(relative_advantage_tich, dim=0), color='m', label="Alg. 1", linestyle=next(linecycler))
    # ax[1].fill_between(x, torch.min(relative_advantage_tich, dim=0)[0].numpy(), \
    #                  y2=torch.max(relative_advantage_tich, dim=0)[0].numpy(), alpha=0.2, color='m')
    # ax[1].plot(x, torch.mean(sel_func_store_tich[:, index_best_tich, :], dim=0), color='m', label="Tichonov")
    # ax[1].fill_between(x, torch.min(sel_func_store_tich[:, index_best_tich, :], dim=0)[0].numpy(), \
    #                    y2=torch.max(sel_func_store_tich[:, index_best_tich, :], dim=0)[0].numpy(), alpha=0.2, color='m')
    ax[1,0].grid(True)
    ax[1,0].set_xscale('log')
    ax[1,0].set_ylabel(r'$\phi - \phi^{\mathrm{FBF}} (\%)$', fontsize=9)
    # ax[1, 0].set_xlabel(r'', fontsize=7)
    fig.text(0.5, 0.04, 'Iteration', ha='center', fontsize=9)
    ax[1,0].set_xlim(1, N_iterations * Steps_between_iterations)

    # Create zoomed plot on the right
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    ax[0,1].plot(x_2, torch.mean(residual_store_hsdm[:, index_hsdm_to_plot, (N_iterations-9*N_iterations//10):], dim=0), color='g', label="HSDM", linestyle=next(linecycler))
    ax[0,1].plot(x_2, torch.mean(residual_store_tich[:, index_tich_to_plot, (N_iterations-9*N_iterations//10):], dim=0), color='m', label="Alg. 1", linestyle=next(linecycler))

    ax[0,1].grid(True, which='minor', axis='both')
    ax[0,1].set_yscale('log')
    ax[0,1].yaxis.tick_right()
    ax[0,1].set_ylim(10**(-4), 10**(-1))
    ax[0,1].set_yticks(np.logspace(-4,-1, 4))
    ax[0, 1].set_yticklabels(np.logspace(-4, -1, 4), minor=True)
    # ax[0,1].set_xlim(1, (N_iterations-N_iterations/10) * Steps_between_iterations)
    # ax[0,1].legend(fontsize=9)

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    relative_advantage_hsdm = 100*(sel_func_store_hsdm[:, index_hsdm_to_plot, :] - sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_hsdm[:, index_hsdm_to_plot, :].size(1)) )\
                              / torch.abs(sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_hsdm[:, index_hsdm_to_plot, :].size(1)))
    ax[1,1].plot(x_2, torch.mean(relative_advantage_hsdm, dim=0)[(N_iterations-9*N_iterations//10):] , color='g', label="HSDM",linestyle=next(linecycler))

    relative_advantage_tich = 100*(sel_func_store_tich[:,index_tich_to_plot,:] - sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, index_tich_to_plot, :].size(1)) )\
                              / torch.abs(sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, index_tich_to_plot, :].size(1)))
    # relative_advantage_tich = sel_func_store_tich[:, index_best_tich, :]
    ax[1,1].plot(x_2, torch.mean(relative_advantage_tich, dim=0)[(N_iterations-9*N_iterations//10):], color='m', label="Alg. 1", linestyle=next(linecycler))
    # ax[1].fill_between(x, torch.min(relative_advantage_tich, dim=0)[0].numpy(), \
    #                  y2=torch.max(relative_advantage_tich, dim=0)[0].numpy(), alpha=0.2, color='m')
    # ax[1].plot(x, torch.mean(sel_func_store_tich[:, index_best_tich, :], dim=0), color='m', label="Tichonov")
    # ax[1].fill_between(x, torch.min(sel_func_store_tich[:, index_best_tich, :], dim=0)[0].numpy(), \
    #                    y2=torch.max(sel_func_store_tich[:, index_best_tich, :], dim=0)[0].numpy(), alpha=0.2, color='m')
    ax[1,1].grid(True)
    ax[1,1].set_xscale('log')
    ax[1,1].yaxis.tick_right()
    # ax[1,1].set_ylim(-500, 200)
    # ax[0, 0].set_yticks(np.arange(-500, 200 + 1, 500))
    # ax[1].set_ylim(-2000, 500)
    # ax[1].set_yticks(np.arange(-2000, 500 + 1, 500))

    plt.tight_layout() ### THESE HAVE TO BE CALLED BEFORE MANUALLY DRAWING THE LINES BETWEEN PLOTS
    fig.tight_layout() ### THESE HAVE TO BE CALLED BEFORE MANUALLY DRAWING THE LINES BETWEEN PLOTS
    fig.subplots_adjust(bottom=0.2)
    ## Draw zoom-in between plots
    ylims_res = ax[0,1].get_ylim()
    xy_1_a = (N_iterations*Steps_between_iterations, torch.mean(residual_store_hsdm[:, index_hsdm_to_plot,-1], dim=0))
    xy_1_b = (N_iterations*Steps_between_iterations,
              torch.mean(residual_store_tich[:, index_tich_to_plot, -1], dim=0))
    xy_2_a = ((N_iterations*Steps_between_iterations)//10, ylims_res[0])
    xy_2_b = ((N_iterations*Steps_between_iterations)//10, ylims_res[1])
    con_a = ConnectionPatch(xyA=xy_1_a, xyB=xy_2_a, coordsA="data", coordsB="data", axesA=ax[0][0], axesB=ax[0][1], color="k")
    con_b = ConnectionPatch(xyA=xy_1_b, xyB=xy_2_b, coordsA="data", coordsB="data", axesA=ax[0][0], axesB=ax[0][1],
                          color="k")
    ax[0][1].add_artist(con_a)
    ax[0][1].add_artist(con_b)

    ylims_res = ax[1,1].get_ylim()
    xy_1_a = (N_iterations*Steps_between_iterations, relative_advantage_hsdm[0, -1])
    xy_1_b = (N_iterations*Steps_between_iterations, relative_advantage_tich[0,-1])
    xy_2_a = ((N_iterations*Steps_between_iterations)//10, ylims_res[0])
    xy_2_b = ((N_iterations*Steps_between_iterations)//10, ylims_res[1])
    con_a = ConnectionPatch(xyA=xy_1_a, xyB=xy_2_a, coordsA="data", coordsB="data", axesA=ax[1][0], axesB=ax[1][1], color="k")
    con_b = ConnectionPatch(xyA=xy_1_b, xyB=xy_2_b, coordsA="data", coordsB="data", axesA=ax[1][0], axesB=ax[1][1],
                          color="k")
    ax[1][1].add_artist(con_a)
    ax[1][1].add_artist(con_b)
    con_a.set_in_layout(False)
    con_b.set_in_layout(False)
    # Create inset
    # lines = ["-", "--", "-.", ":"]
    # linecycler = cycle(lines)
    # axins = ax[1].inset_axes([10**5, -1000, 5*10**5, 500])
    # # axins = ax[1].inset_axes([100, -10, 5 * 10, 50])
    # # subregion of the original image
    # x1, x2, y1, y2 = 9 * 10 ** 4, 10 ** 5, -600, 0
    # # x1, x2, y1, y2 = 90, 100, -60, 0
    # x_inset = range(x1*Steps_between_iterations, Steps_between_iterations * x2, Steps_between_iterations)
    # axins.plot(x_inset, torch.mean(relative_advantage_hsdm, dim=0)[x1:x2], color='g', label="HSDM", linestyle=next(linecycler))
    # axins.plot(x_inset, torch.mean(relative_advantage_tich, dim=0)[x1:x2], color='m', label="Alg. 1", linestyle=next(linecycler))

    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])

    # ax[1].indicate_inset_zoom(axins, edgecolor="black")

    plt.draw()

    plt.show(block=False)

    fig.savefig(directory + '/Figures/Method_comparison_HSDM_tich_FBF.png')
    fig.savefig(directory + '/Figures/Method_comparison_HSDM_tich_FBF.pdf')


    ################################################
    ##### Plots relative to Tichonov parameters#####
    ################################################

    all_indexes_with_approx_reg_and_weight_reg_to_plot = []
    all_indexes_with_weight_reg_and_alpha_to_plot = []
    all_indexes_with_approx_reg_to_plot_and_alpha_to_plot = []

    for i in range(len(parameters_tested_tich)):
        if parameters_tested_tich[i][0]==weight_reg_to_plot and parameters_tested_tich[i][1]==alpha_to_plot:
            all_indexes_with_weight_reg_and_alpha_to_plot.append(i)
        if parameters_tested_tich[i][0] == weight_reg_to_plot and parameters_tested_tich[i][2] == approx_reg_to_plot:
            all_indexes_with_approx_reg_and_weight_reg_to_plot.append(i)
        if parameters_tested_tich[i][1]==alpha_to_plot and parameters_tested_tich[i][2] == approx_reg_to_plot:
            all_indexes_with_approx_reg_to_plot_and_alpha_to_plot.append(i)
    ## Plot 2.1: fix regularization decay and alpha, vary epsilon decay

    fig, ax = plt.subplots(2, 2, figsize=(4, 2.1), sharex='col', width_ratios=[4, 1])
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for i in all_indexes_with_approx_reg_to_plot_and_alpha_to_plot:
        ax[0,0].plot(x_1, torch.mean(residual_store_tich[:, i, :], dim=0), label=str(parameters_tested_tich[i][0]), linestyle=next(linecycler))
        print("Final residual for xi_gamma=" + str(parameters_tested_tich[i][0]) + " :" + str(torch.mean(residual_store_tich[:, i, :], dim=0)[-1]))
    ax[0,0].grid(True, which='both', axis='both')
    ax[0,0].set_yscale('log')
    ax[0,0].set_xscale('log')
    ax[0,0].set_ylabel("Residual", fontsize=9)
    ax[0,0].set_xlim(1, N_iterations* Steps_between_iterations)
    ax[0,0].set_ylim(10 ** (-4), 10 ** (2))
    ax[0,0].set_yticks([10**(-4), 10 ** (-2), 1, 10**(2)])
    ax[0,0].legend(title=r'$\xi$', fontsize=7)

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for i in all_indexes_with_approx_reg_to_plot_and_alpha_to_plot:
        values_to_plot = 100*(sel_func_store_tich[:, i, :] - sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, i, :].size(1))) \
                                       / torch.abs(sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, i, :].size(1))) #dimensions wont work here
        ax[1,0].plot(x_1, torch.mean(values_to_plot, dim=0), label=str(parameters_tested_tich[i][0]), linestyle=next(linecycler))
        print("Final sel.fun. advantage for xi_gamma=" + str(parameters_tested_tich[i][0]) + " :" + str(torch.mean(values_to_plot, dim=0)[-1]))
    ax[1,0].grid(True)
    ax[1,0].set_xscale('log')
    ax[1,0].set_ylabel(r'$\phi - \phi^{\mathrm{FBF}} (\%)$', fontsize=9)
    fig.text(0.5, 0.04, 'Iteration', ha='center', fontsize=9)
    ax[1,0].set_xlim(1, N_iterations * Steps_between_iterations)

    # Create zoomed plot on the right
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for i in all_indexes_with_approx_reg_to_plot_and_alpha_to_plot:
        ax[0,1].plot(x_2, torch.mean(residual_store_tich[:, i, (N_iterations-9*N_iterations//10):], dim=0), label=str(parameters_tested_tich[i][0]), linestyle=next(linecycler))
        print("Final residual for xi_gamma=" + str(parameters_tested_tich[i][0]) + " :" + str(torch.mean(residual_store_tich[:, i, :], dim=0)[-1]))
    ax[0,1].grid(True, which='minor', axis='both')
    ax[0,1].set_yscale('log')
    ax[0,1].yaxis.tick_right()
    ax[0, 1].set_ylim(10 ** (-4), 10 ** (-1))
    ax[0, 1].set_yticks(np.logspace(-4, -1, 4))
    ax[0, 1].set_yticklabels(np.logspace(-4, -1, 4), minor=True)

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for i in all_indexes_with_approx_reg_to_plot_and_alpha_to_plot:
        values_to_plot = 100*(sel_func_store_tich[:, i, :] - sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, i, :].size(1))) \
                                       / torch.abs(sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, i, :].size(1))) #dimensions wont work here
        ax[1,1].plot(x_2, torch.mean(values_to_plot, dim=0)[(N_iterations-9*N_iterations//10):], label=str(parameters_tested_tich[i][0]), linestyle=next(linecycler))
        print("Final sel.fun. advantage for xi_gamma=" + str(parameters_tested_tich[i][0]) + " :" + str(torch.mean(values_to_plot, dim=0)[-1]))
    ax[1,1].grid(True)
    ax[1,1].set_xscale('log')
    ax[1,1].yaxis.tick_right()
    # ax[1,1].set_ylim(-500, 200)

    plt.tight_layout() ### THESE HAVE TO BE CALLED BEFORE MANUALLY DRAWING THE LINES BETWEEN PLOTS
    fig.tight_layout() ### THESE HAVE TO BE CALLED BEFORE MANUALLY DRAWING THE LINES BETWEEN PLOTS
    fig.subplots_adjust(bottom=0.2)
    ## Draw zoom-in between plots
    ylims_res = ax[0,1].get_ylim()
    xy_1_a = (N_iterations*Steps_between_iterations, \
              torch.min(torch.mean(residual_store_tich[:, all_indexes_with_approx_reg_to_plot_and_alpha_to_plot,(N_iterations - 9 * N_iterations // 10)], dim=0)))
    xy_1_b = ((N_iterations*Steps_between_iterations),
              torch.max(torch.mean(residual_store_tich[:, all_indexes_with_approx_reg_to_plot_and_alpha_to_plot,(N_iterations - 9 * N_iterations // 10)], dim=0)))
    xy_2_a = (N_iterations*Steps_between_iterations // 10, ylims_res[0])
    xy_2_b = ((N_iterations*Steps_between_iterations) // 10, ylims_res[1])
    con_a = ConnectionPatch(xyA=xy_1_a, xyB=xy_2_a, coordsA="data", coordsB="data", axesA=ax[0][0], axesB=ax[0][1], color="k")
    con_b = ConnectionPatch(xyA=xy_1_b, xyB=xy_2_b, coordsA="data", coordsB="data", axesA=ax[0][0], axesB=ax[0][1],
                          color="k")
    ax[0][1].add_artist(con_a)
    ax[0][1].add_artist(con_b)

    ylims_res = ax[1,1].get_ylim()
    xy_1_a = (N_iterations*Steps_between_iterations, relative_advantage_tich[0, -1])
    xy_1_b = ((N_iterations*Steps_between_iterations), relative_advantage_tich[0,-1])
    xy_2_a = (N_iterations*Steps_between_iterations // 10, ylims_res[0])
    xy_2_b = ((N_iterations*Steps_between_iterations) // 10, ylims_res[1])
    con_a = ConnectionPatch(xyA=xy_1_a, xyB=xy_2_a, coordsA="data", coordsB="data", axesA=ax[1][0], axesB=ax[1][1], color="k")
    con_b = ConnectionPatch(xyA=xy_1_b, xyB=xy_2_b, coordsA="data", coordsB="data", axesA=ax[1][0], axesB=ax[1][1],
                          color="k")
    ax[1][1].add_artist(con_a)
    ax[1][1].add_artist(con_b)
    con_a.set_in_layout(False)
    con_b.set_in_layout(False)
    plt.show(block=False)
    fig.savefig(directory + '/Figures/Tikhonov_weight_comparison.png')
    fig.savefig(directory + '/Figures/Tikhonov_weight_comparison.pdf')

    ## Plot 2.2: fix regularization decay epsilon decay, vary alpha

    fig, ax = plt.subplots(2, 2, figsize=(4, 2.1), sharex='col', width_ratios=[4, 1])
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for i in all_indexes_with_approx_reg_and_weight_reg_to_plot:
        ax[0,0].plot(x_1, torch.mean(residual_store_tich[:, i, :], dim=0), label=str(parameters_tested_tich[i][1]), linestyle=next(linecycler))
        print("Final residual for alpha=" + str(parameters_tested_tich[i][1]) + " :" + str(torch.mean(residual_store_tich[:, i, :], dim=0)[-1]))
    ax[0,0].grid(True, which='both', axis='both')
    ax[0,0].set_yscale('log')
    ax[0,0].set_xscale('log')
    ax[0,0].set_ylabel("Residual", fontsize=9)
    ax[0,0].set_xlim(1, N_iterations* Steps_between_iterations)
    ax[0,0].set_ylim(10 ** (-4), 10 ** (2))
    ax[0,0].set_yticks([10**(-4), 10 ** (-2), 1, 10**(2)])
    ax[0,0].legend(title=r'$\alpha$', fontsize=7)

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for i in all_indexes_with_approx_reg_and_weight_reg_to_plot:
        values_to_plot = 100*(sel_func_store_tich[:, i, :] - sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, i, :].size(1))) \
                                       / torch.abs(sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, i, :].size(1))) #dimensions wont work here
        ax[1,0].plot(x_1, torch.mean(values_to_plot, dim=0), label=str(parameters_tested_tich[i][1]), linestyle=next(linecycler))
        print("Final sel.fun. advantage for alpha=" + str(parameters_tested_tich[i][1]) + " :" + str(torch.mean(values_to_plot, dim=0)[-1]))
    ax[1,0].grid(True)
    ax[1,0].set_xscale('log')
    ax[1,0].set_ylabel(r'$\phi - \phi^{\mathrm{FBF}} (\%)$', fontsize=9)
    fig.text(0.5, 0.04, 'Iteration', ha='center', fontsize=9)
    ax[1,0].set_xlim(1, N_iterations * Steps_between_iterations)

    # Create zoomed plot on the right
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for i in all_indexes_with_approx_reg_and_weight_reg_to_plot:
        ax[0,1].plot(x_2, torch.mean(residual_store_tich[:, i, (N_iterations-9*N_iterations//10):], dim=0), label=str(parameters_tested_tich[i][1]), linestyle=next(linecycler))
        print("Final residual for alpha=" + str(parameters_tested_tich[i][1]) + " :" + str(torch.mean(residual_store_tich[:, i, :], dim=0)[-1]))
    ax[0,1].grid(True, which='minor', axis='both')
    ax[0,1].set_yscale('log')
    ax[0,1].yaxis.tick_right()
    ax[0, 1].set_ylim(10 ** (-4), 10 ** (-1))
    ax[0, 1].set_yticks(np.logspace(-4, -1, 4))
    ax[0, 1].set_yticklabels(np.logspace(-4, -1, 4), minor=True)

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for i in all_indexes_with_approx_reg_and_weight_reg_to_plot:
        values_to_plot = 100*(sel_func_store_tich[:, i, :] - sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, i, :].size(1))) \
                                       / torch.abs(sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, i, :].size(1))) #dimensions wont work here
        ax[1,1].plot(x_2, torch.mean(values_to_plot, dim=0)[(N_iterations-9*N_iterations//10):], label=str(parameters_tested_tich[i][1]), linestyle=next(linecycler))
        print("Final sel.fun. advantage for alpha=" + str(parameters_tested_tich[i][1]) + " :" + str(torch.mean(values_to_plot, dim=0)[-1]))
    ax[1,1].grid(True)
    ax[1,1].set_xscale('log')
    ax[1,1].yaxis.tick_right()
    # ax[1,1].set_ylim(-500, 200)

    plt.tight_layout() ### THESE HAVE TO BE CALLED BEFORE MANUALLY DRAWING THE LINES BETWEEN PLOTS
    fig.tight_layout() ### THESE HAVE TO BE CALLED BEFORE MANUALLY DRAWING THE LINES BETWEEN PLOTS
    fig.subplots_adjust(bottom=0.2)
    ## Draw zoom-in between plots
    ylims_res = ax[0,1].get_ylim()
    xy_1_a = (N_iterations*Steps_between_iterations, \
              torch.min(torch.mean(residual_store_tich[:, all_indexes_with_approx_reg_to_plot_and_alpha_to_plot,(N_iterations - 9 * N_iterations // 10)], dim=0)))
    xy_1_b = ((N_iterations*Steps_between_iterations),
              torch.max(torch.mean(residual_store_tich[:, all_indexes_with_approx_reg_to_plot_and_alpha_to_plot,(N_iterations - 9 * N_iterations // 10)], dim=0)))
    xy_2_a = (N_iterations*Steps_between_iterations // 10, ylims_res[0])
    xy_2_b = ((N_iterations*Steps_between_iterations) // 10, ylims_res[1])
    con_a = ConnectionPatch(xyA=xy_1_a, xyB=xy_2_a, coordsA="data", coordsB="data", axesA=ax[0][0], axesB=ax[0][1], color="k")
    con_b = ConnectionPatch(xyA=xy_1_b, xyB=xy_2_b, coordsA="data", coordsB="data", axesA=ax[0][0], axesB=ax[0][1],
                          color="k")
    ax[0][1].add_artist(con_a)
    ax[0][1].add_artist(con_b)

    ylims_res = ax[1,1].get_ylim()
    xy_1_a = (N_iterations*Steps_between_iterations, relative_advantage_tich[0, -1])
    xy_1_b = ((N_iterations*Steps_between_iterations), relative_advantage_tich[0,-1])
    xy_2_a = (N_iterations*Steps_between_iterations // 10, ylims_res[0])
    xy_2_b = ((N_iterations*Steps_between_iterations) // 10, ylims_res[1])
    con_a = ConnectionPatch(xyA=xy_1_a, xyB=xy_2_a, coordsA="data", coordsB="data", axesA=ax[1][0], axesB=ax[1][1], color="k")
    con_b = ConnectionPatch(xyA=xy_1_b, xyB=xy_2_b, coordsA="data", coordsB="data", axesA=ax[1][0], axesB=ax[1][1],
                          color="k")
    ax[1][1].add_artist(con_a)
    ax[1][1].add_artist(con_b)
    con_a.set_in_layout(False)
    con_b.set_in_layout(False)
    plt.show(block=False)
    fig.savefig(directory + '/Figures/Tikhonov_alpha_comparison.png')
    fig.savefig(directory + '/Figures/Tikhonov_alpha_comparison.pdf')

    ## Plot 2.3: fix weight decay and alpha, vary approx. decay

    fig, ax = plt.subplots(2, 2, figsize=(4, 2.1), sharex='col', width_ratios=[4, 1])
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for i in all_indexes_with_weight_reg_and_alpha_to_plot:
        ax[0, 0].plot(x_1, torch.mean(residual_store_tich[:, i, :], dim=0), label=str(parameters_tested_tich[i][2]),
                      linestyle=next(linecycler))
        print("Final residual for xi_eps/xi_gamma=" + str(parameters_tested_tich[i][2]) + " :" + str(
            torch.mean(residual_store_tich[:, i, :], dim=0)[-1]))
    ax[0, 0].grid(True, which='both', axis='both')
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_ylabel("Residual", fontsize=9)
    ax[0, 0].set_xlim(1, N_iterations * Steps_between_iterations)
    ax[0, 0].set_ylim(10 ** (-4), 10 ** (2))
    ax[0, 0].set_yticks([10 ** (-4), 10 ** (-2), 1, 10 ** (2)])
    ax[0, 0].legend(title=r'$\zeta$', fontsize=7)

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for i in all_indexes_with_weight_reg_and_alpha_to_plot:
        values_to_plot = 100 * (sel_func_store_tich[:, i, :] - sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,
            sel_func_store_tich[:, i,:].size(1))) \
            / torch.abs(sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1, sel_func_store_tich[:, i, :].size(1)))  # dimensions wont work here
        ax[1, 0].plot(x_1, torch.mean(values_to_plot, dim=0), label=str(parameters_tested_tich[i][2]),
                      linestyle=next(linecycler))
        print("Final sel.fun. advantage for xi_eps/xi_gamma=" + str(parameters_tested_tich[i][1]) + " :" + str(
            torch.mean(values_to_plot, dim=0)[-1]))
    ax[1, 0].grid(True)
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_ylabel(r'$\phi - \phi^{\mathrm{FBF}} (\%)$', fontsize=9)
    fig.text(0.5, 0.04, 'Iteration', ha='center', fontsize=9)
    ax[1, 0].set_xlim(1, N_iterations * Steps_between_iterations)

    # Create zoomed plot on the right
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for i in all_indexes_with_weight_reg_and_alpha_to_plot:
        ax[0, 1].plot(x_2, torch.mean(residual_store_tich[:, i, (N_iterations - 9 * N_iterations // 10):], dim=0),
                      label=str(parameters_tested_tich[i][2]), linestyle=next(linecycler))
        print("Final residual for xi_eps/xi_gamma=" + str(parameters_tested_tich[i][2]) + " :" + str(
            torch.mean(residual_store_tich[:, i, :], dim=0)[-1]))
    ax[0, 1].grid(True, which='minor', axis='both')
    ax[0, 1].set_yscale('log')
    ax[0, 1].yaxis.tick_right()
    ax[0, 1].set_ylim(10 ** (-4), 10 ** (-1))
    ax[0, 1].set_yticks(np.logspace(-4, -1, 4))
    ax[0, 1].set_yticklabels(np.logspace(-4, -1, 4), minor=True)

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for i in all_indexes_with_weight_reg_and_alpha_to_plot:
        values_to_plot = 100 * (sel_func_store_tich[:, i, :] - sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,
            sel_func_store_tich[:, i,:].size(1))) \
             / torch.abs(sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, i, :].size(1)))  # dimensions wont work here
        ax[1, 1].plot(x_2, torch.mean(values_to_plot, dim=0)[(N_iterations - 9 * N_iterations // 10):],
                      label=str(parameters_tested_tich[i][2]), linestyle=next(linecycler))
        print("Final sel.fun. advantage for xi_eps/xi_gamma=" + str(parameters_tested_tich[i][2]) + " :" + str(
            torch.mean(values_to_plot, dim=0)[-1]))
    ax[1, 1].grid(True)
    ax[1, 1].set_xscale('log')
    ax[1, 1].yaxis.tick_right()
    # ax[1,1].set_ylim(-500, 200)


    plt.tight_layout()  ### THESE HAVE TO BE CALLED BEFORE MANUALLY DRAWING THE LINES BETWEEN PLOTS
    fig.tight_layout()  ### THESE HAVE TO BE CALLED BEFORE MANUALLY DRAWING THE LINES BETWEEN PLOTS
    fig.subplots_adjust(bottom=0.2)
    ## Draw zoom-in between plots
    ylims_res = ax[0, 1].get_ylim()
    xy_1_a = (N_iterations * Steps_between_iterations, \
              torch.min(torch.mean(residual_store_tich[:, all_indexes_with_weight_reg_and_alpha_to_plot,
                                   (N_iterations - 9 * N_iterations // 10)], dim=0)))
    xy_1_b = ((N_iterations * Steps_between_iterations),
              torch.max(torch.mean(residual_store_tich[:, all_indexes_with_weight_reg_and_alpha_to_plot,
                                   (N_iterations - 9 * N_iterations // 10)], dim=0)))
    xy_2_a = (N_iterations * Steps_between_iterations // 10, ylims_res[0])
    xy_2_b = ((N_iterations * Steps_between_iterations) // 10, ylims_res[1])
    con_a = ConnectionPatch(xyA=xy_1_a, xyB=xy_2_a, coordsA="data", coordsB="data", axesA=ax[0][0], axesB=ax[0][1],
                            color="k")
    con_b = ConnectionPatch(xyA=xy_1_b, xyB=xy_2_b, coordsA="data", coordsB="data", axesA=ax[0][0], axesB=ax[0][1],
                            color="k")
    ax[0][1].add_artist(con_a)
    ax[0][1].add_artist(con_b)

    ylims_res = ax[1, 1].get_ylim()
    xy_1_a = (N_iterations * Steps_between_iterations, relative_advantage_tich[0, -1])
    xy_1_b = ((N_iterations * Steps_between_iterations), relative_advantage_tich[0, -1])
    xy_2_a = (N_iterations * Steps_between_iterations // 10, ylims_res[0])
    xy_2_b = ((N_iterations * Steps_between_iterations) // 10, ylims_res[1])
    con_a = ConnectionPatch(xyA=xy_1_a, xyB=xy_2_a, coordsA="data", coordsB="data", axesA=ax[1][0], axesB=ax[1][1],
                            color="k")
    con_b = ConnectionPatch(xyA=xy_1_b, xyB=xy_2_b, coordsA="data", coordsB="data", axesA=ax[1][0], axesB=ax[1][1],
                            color="k")
    ax[1][1].add_artist(con_a)
    ax[1][1].add_artist(con_b)
    con_a.set_in_layout(False)
    con_b.set_in_layout(False)
    plt.show(block=False)
    # legend_entries = []
    # for eps in set(df.loc[(df['Method'] == TICH) & (df[r'$\alpha$'] == alpha_to_plot) & (
    #         df['Reg. decay'] == weight_reg_to_plot)]['Approx. decay'].tolist()):
    #     legend_entries.append(str(eps * weight_reg_to_plot))
    # g.legend(legend_entries, title=r'$\xi_{\varepsilon}$', fontsize=9)
    fig.savefig(directory + '/Figures/Tikhonov_epsilon_comparison.png',dpi=fig.dpi, bbox_inches='tight')
    fig.savefig(directory + '/Figures/Tikhonov_epsilon_comparison.pdf',dpi=fig.dpi, bbox_inches='tight')


    ### DEPRECATED PLOTS
    #
    # fig, ax = plt.subplots(2, 1, figsize=(4, 1.9), layout='constrained', sharex='col')
    # x = range(1, Steps_between_iterations * N_iterations, Steps_between_iterations)
    # lines = ["-", "--", "-.", ":"]
    # linecycler = cycle(lines)
    # for i in all_indexes_with_approx_reg_to_plot_and_alpha_to_plot:
    #     ax[0].plot(x, torch.mean(residual_store_tich[:, i, :], dim=0), label=str(parameters_tested_tich[i][0]), linestyle=next(linecycler))
    #     print("Final residual for xi_gamma=" + str(parameters_tested_tich[i][0]) + " :" + str(torch.mean(residual_store_tich[:, i, :], dim=0)[-1]))
    # ax[0].grid(True)
    # ax[0].set_yscale('log')
    # ax[0].set_xscale('log')
    # # ax[0].set_title("Tichonov")
    # ax[0].set_ylabel("Residual", fontsize=9)
    # ax[0].set_xlim(1, N_iterations * 10)
    # ax[0].set_ylim(ax[0].get_ylim()[0] / 2, ax[0].get_ylim()[1] * 2)
    # lines = ["-", "--", "-.", ":"]
    # linecycler = cycle(lines)
    # for i in all_indexes_with_approx_reg_to_plot_and_alpha_to_plot:
    #     values_to_plot = 100*(sel_func_store_tich[:, i, :] - sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, i, :].size(1))) \
    #                                    / torch.abs(sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, i, :].size(1))) #dimensions wont work here
    #     ax[1].plot(x, torch.mean(values_to_plot, dim=0), label=str(parameters_tested_tich[i][0]), linestyle=next(linecycler))
    #     print("Final sel.fun. advantage for xi_gamma=" + str(parameters_tested_tich[i][0]) + " :" + str(torch.mean(values_to_plot, dim=0)[-1]))
    # ax[1].grid(True)
    # ax[1].set_xscale('log')
    # # ax[1,0].set_yscale('log')
    # ax[1].set_ylabel(r'$\phi - \phi^{\mathrm{FBF}} (\%)$', fontsize=9)
    # ax[1].set_xlabel('Iteration', fontsize=9)
    # ax[1].set_xlim(1, N_iterations * 10)
    # # ax[1].set_ylim(-2000, 500)
    # # ax[1].set_yticks(np.arange(-2000, 500 + 1, 500))
    # # g.legend(labels = parameters_labels_tich)
    # ax[1].legend(title=r'$\xi_{\gamma}$', fontsize=7) #might not work
    # plt.show(block=False)
    # fig.savefig(directory + '/Figures/Tikhonov_weight_comparison.png')
    # fig.savefig(directory + '/Figures/Tikhonov_weight_comparison.pdf')
    #
    # ### Plot 2.2: fix weight decay and approx decay, vary alpha
    # fig, ax = plt.subplots(2, 1, figsize=(4, 1.9), layout='constrained', sharex='col')
    # x = range(1, Steps_between_iterations * N_iterations, Steps_between_iterations)
    # lines = ["-", "--", "-.", ":"]
    # linecycler = cycle(lines)
    # for i in all_indexes_with_approx_reg_and_weight_reg_to_plot:
    #     ax[0].plot(x, torch.mean(residual_store_tich[:, i, :], dim=0), label=str(parameters_tested_tich[i][1]), linestyle=next(linecycler))
    #     print("Final residual for alpha=" + str(parameters_tested_tich[i][1]) + " :" + str(
    #         torch.mean(residual_store_tich[:, i, :], dim=0)[-1]))
    # ax[0].grid(True)
    # ax[0].set_yscale('log')
    # ax[0].set_xscale('log')
    # # ax[0].set_title("Tichonov")
    # ax[0].set_ylabel("Residual", fontsize=9)
    # ax[0].set_xlim(1, N_iterations * 10)
    # ax[0].set_ylim(ax[0].get_ylim()[0] / 2, ax[0].get_ylim()[1] * 2)
    # lines = ["-", "--", "-.", ":"]
    # linecycler = cycle(lines)
    # for i in all_indexes_with_approx_reg_and_weight_reg_to_plot:
    #     values_to_plot = 100*(sel_func_store_tich[:, i, :] - sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, i, :].size(1))) \
    #                                    / torch.abs(sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, i, :].size(1))) #dimensions wont work here
    #     ax[1].plot(x, torch.mean(values_to_plot, dim=0), label=str(parameters_tested_tich[i][1]), linestyle=next(linecycler))
    #     print("Final sel.fun. advantage for alpha=" + str(parameters_tested_tich[i][1]) + " :" + str(
    #         torch.mean(values_to_plot, dim=0)[-1]))
    # ax[1].grid(True)
    # ax[1].set_xscale('log')
    # # ax[1,0].set_yscale('log')
    # ax[1].set_ylabel(r'$\phi - \phi^{\mathrm{FBF}} (\%)$', fontsize=9)
    # ax[1].set_xlabel('Iteration', fontsize=9)
    # ax[1].set_xlim(1, N_iterations * 10)
    # # ax[1].set_ylim(-2000, 500)
    # # ax[1].set_yticks(np.arange(-2000, 500 + 1, 500))
    # # g.legend(labels = parameters_labels_tich)
    # ax[1].legend(title=r'$\alpha$', fontsize=7)
    # plt.show(block=False)
    # fig.savefig(directory + '/Figures/Tikhonov_alpha_comparison.png')
    # fig.savefig(directory + '/Figures/Tikhonov_alpha_comparison.pdf')
    #
    # ### Plot 2.3: fix weight decay and alpha, vary approx. decay
    # fig, ax = plt.subplots(2, 1, figsize=(4, 1.9), layout='constrained', sharex='col')
    # lines = ["-", "--", "-.", ":"]
    # linecycler = cycle(lines)
    # x = range(1, Steps_between_iterations * N_iterations, Steps_between_iterations)
    # for i in all_indexes_with_weight_reg_and_alpha_to_plot:
    #     ax[0].plot(x, torch.mean(residual_store_tich[:, i, :], dim=0), label=str(parameters_tested_tich[i][2]), linestyle=next(linecycler))
    #     print("Final residual for xi_eps/xi_gamma=" + str(parameters_tested_tich[i][2]) + " :" + str(
    #         torch.mean(residual_store_tich[:, i, :], dim=0)[-1]))
    # ax[0].grid(True)
    # ax[0].set_yscale('log')
    # ax[0].set_xscale('log')
    # # ax[0].set_title("Tichonov")
    # ax[0].set_ylabel("Residual", fontsize=9)
    # ax[0].set_xlim(1, N_iterations * 10)
    # ax[0].set_ylim(ax[0].get_ylim()[0] / 2, ax[0].get_ylim()[1] * 2)
    # lines = ["-", "--", "-.", ":"]
    # linecycler = cycle(lines)
    # for i in all_indexes_with_weight_reg_and_alpha_to_plot:
    #     values_to_plot = 100*(sel_func_store_tich[:, i, :] - sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, i, :].size(1))) \
    #                                    / torch.abs(sel_func_store_std[:, 0, -1].unsqueeze(1).repeat(1,sel_func_store_tich[:, i, :].size(1))) #dimensions wont work here
    #     ax[1].plot(x, torch.mean(values_to_plot, dim=0), label=str(parameters_tested_tich[i][2]), linestyle=next(linecycler))
    #     print("Final sel.fun. advantage for xi_eps/xi_gamma=" + str(parameters_tested_tich[i][2]) + " :" + str(
    #         torch.mean(values_to_plot, dim=0)[-1]))
    # ax[1].grid(True)
    # ax[1].set_xscale('log')
    # # ax[1,0].set_yscale('log')
    # ax[1].set_ylabel(r'$\phi - \phi^{\mathrm{FBF}} (\%)$', fontsize=9)
    # ax[1].set_xlabel('Iteration', fontsize=9)
    # ax[1].set_xlim(1, N_iterations * 10)
    # # ax[1].set_ylim(-2000, 500)
    # # ax[1].set_yticks(np.arange(-2000, 500 + 1, 500))
    # # g.legend(labels = parameters_labels_tich)
    # ax[1].legend(title=r'$\xi_{\varepsilon}/\xi_{\gamma}$', fontsize=7)
    # plt.show(block=False)
    # # legend_entries = []
    # # for eps in set(df.loc[(df['Method'] == TICH) & (df[r'$\alpha$'] == alpha_to_plot) & (
    # #         df['Reg. decay'] == weight_reg_to_plot)]['Approx. decay'].tolist()):
    # #     legend_entries.append(str(eps * weight_reg_to_plot))
    # # g.legend(legend_entries, title=r'$\xi_{\varepsilon}$', fontsize=9)
    # fig.savefig(directory + '/Figures/Tikhonov_epsilon_comparison.png',dpi=fig.dpi, bbox_inches='tight')
    # fig.savefig(directory + '/Figures/Tikhonov_epsilon_comparison.pdf',dpi=fig.dpi, bbox_inches='tight')
#
#     N_datapoints = N_tests * (N_tested_sets_of_params_tich + N_tested_sets_of_params_hsdm + 1)  * N_iterations
#     list_residual = np.zeros((N_datapoints,1))
#     list_iteration = np.zeros((N_datapoints,1))
#     list_method = np.zeros((N_datapoints, 1))
#     list_exponent_regularization = np.zeros((N_datapoints,1))
#     list_inertia_tich = np.zeros((N_datapoints, 1))
#     list_decay_approx = np.zeros((N_datapoints, 1))
#     list_sel_fun = np.zeros((N_datapoints,1))
#     list_relative_advantage = np.zeros((N_datapoints,1))
#
#     index_datapoint=0
#     for test in range(N_tests):
#         for par_index in range(N_tested_sets_of_params_tich):
#             for iteration in range(N_iterations):
#                 # Tichonov
#                 list_residual[index_datapoint]=residual_store_tich[test,par_index,iteration].item()
#                 list_iteration[index_datapoint] = iteration * 10 # residual is sampled every 10 iterations
#                 list_method[index_datapoint] = TICH
#                 list_exponent_regularization[index_datapoint] = parameters_tested_tich[par_index][0]
#                 list_inertia_tich[index_datapoint] = parameters_tested_tich[par_index][1]
#                 list_decay_approx[index_datapoint] = parameters_tested_tich[par_index][2]
#                 list_sel_fun[index_datapoint] = sel_func_store_tich[test,par_index,iteration].item()
#                 list_relative_advantage[index_datapoint] = 100*(sel_func_store_tich[test, par_index, iteration] - sel_func_store_std[test, 0, -1]) \
#                                                            / torch.abs(sel_func_store_std[test, 0, -1])
#                 index_datapoint = index_datapoint +1
#     for test in range(N_tests):
#         for par_index in range(N_tested_sets_of_params_hsdm):
#             for iteration in range(N_iterations):
#                 # HSDM
#                 list_residual[index_datapoint] = residual_store_hsdm[test, par_index, iteration]
#                 list_iteration[index_datapoint] = iteration * 10 # residual is sampled every 10 iterations
#                 list_method[index_datapoint] = HSDM
#                 list_exponent_regularization[index_datapoint] = parameters_tested_hsdm[par_index][0]
#                 list_inertia_tich[index_datapoint] = None
#                 list_decay_approx[index_datapoint] = None
#                 list_sel_fun[index_datapoint] = sel_func_store_hsdm[test, par_index, iteration].item()
#                 list_relative_advantage[index_datapoint] = 100*(sel_func_store_hsdm[test, par_index, iteration] -
#                                                             sel_func_store_std[test, 0, -1]) \
#                                                            / torch.abs(sel_func_store_std[test, 0, -1])
#                 index_datapoint = index_datapoint + 1
#     for test in range(N_tests):
#         for iteration in range(N_iterations):
#             # FBF
#             list_residual[index_datapoint] = residual_store_std[test, 0, iteration]
#             list_iteration[index_datapoint] = iteration * 10 # residual is sampled every 10 iterations
#             list_method[index_datapoint] = FBF
#             list_exponent_regularization[index_datapoint] = None
#             list_inertia_tich[index_datapoint] = None
#             list_decay_approx[index_datapoint] = None
#             list_sel_fun[index_datapoint] = sel_func_store_std[test, 0, iteration].item()
#             index_datapoint = index_datapoint +1
#     df = pd.DataFrame({'Reg. decay': list_exponent_regularization[:,0],r'$\alpha$': list_inertia_tich[:,0], 'Approx. decay': list_decay_approx[:,0],\
#                                     'Residual': list_residual[:,0], 'Iteration': list_iteration[:,0],'Method': list_method[:,0],\
#                                     'Sel. Fun.': list_sel_fun[:,0], 'Rel. advantage':list_relative_advantage[:,0] })
#     # df['Parameters set'] = df['Parameters set'].map(parameters_labels)
#
#     # Save dataframe
#     f = open('saved_dataframe.pkl', 'wb')
#     pickle.dump([df, parameters_tested_tich, parameters_tested_hsdm, N_iterations], f)
#     f.close()
#     print("DataFrame file created.")
# else:
#     f = open('saved_dataframe.pkl', 'rb')
#     df, parameters_tested_tich, parameters_tested_hsdm, N_iterations = pickle.load(f)
#     f.close()
# ##########
# ## PLOT ##
# ##########
# # create dictionary that maps from parameter set to label
# print("DataFrame acquired. Plotting...")
#
#
# parameters_labels_tich = []
# for i in range(len(parameters_tested_tich)):
#     parameters_labels_tich.append(r"$a$ = " + str(parameters_tested_tich[i][0]) +  # r"; $\alpha$ = " + str(parameters_tested[i][1]) + \
#             r"; $b$ = " + str(parameters_tested_tich[i][0]*parameters_tested_tich[i][2]))
#
# parameters_labels_hsdm = []
# for i in range(len(parameters_tested_hsdm)):
#     parameters_labels_hsdm.append(r"$a$ = " + str(parameters_tested_hsdm[i][0]))
#
#
# # Define which values to keep fixed while plotting the others
# weight_reg_to_plot = .6
# approx_reg_to_plot = 2
# alpha_to_plot = 1.
# #### Plot 1: fix epsilon decay and alpha, vary regularization decay
# fig, ax = plt.subplots(2,1, figsize=(4 * 1, 2.2 * 1), layout='constrained', sharex='col')
# g = sns.lineplot(data=df.loc[(df['Method']==TICH) & (df[r'$\alpha$']==alpha_to_plot) & (df['Approx. decay']==approx_reg_to_plot)],\
#                  drawstyle='steps-pre', errorbar=None, \
#                  estimator='mean', x='Iteration', palette='bright',
#                  y='Residual', hue='Reg. decay', style='Reg. decay', legend=False, linewidth = 2.0, ax=ax[0] )
# ax[0].grid(True)
# ax[0].set_yscale('log')
# ax[0].set_xscale('log')
# # ax[0].set_title("Tichonov")
# ax[0].set_ylabel("Residual", fontsize = 9)
# ax[0].set_xlabel('Iteration', fontsize = 9)
# ax[0].set_xlim(1, N_iterations * 10 )
# ax[0].set_ylim(ax[0].get_ylim()[0]/2, ax[0].get_ylim()[1]*2 )
# # g.legend(labels = parameters_labels_tich)
# g = sns.lineplot(data=df.loc[(df['Method']==TICH) & (df[r'$\alpha$']==alpha_to_plot) & (df['Approx. decay']==approx_reg_to_plot)], drawstyle='steps-pre', errorbar=None, \
#                  estimator='mean', x='Iteration', palette='bright',
#                  y='Rel. advantage', hue='Reg. decay', style='Reg. decay', linewidth = 2.0, ax=ax[1])
# ax[1].grid(True)
# ax[1].set_xscale('log')
# # ax[1,0].set_yscale('log')
# ax[1].set_ylabel(r'$\phi - \phi^{\mathrm{FBF}} (\%)$', fontsize = 9)
# ax[1].set_xlabel('Iteration', fontsize = 9)
# ax[1].set_xlim(1, N_iterations * 10 )
# ax[1].set_ylim(-1000, 0)
# ax[1].set_yticks(np.arange(-1000, 0 + 1, 500))
# # g.legend(labels = parameters_labels_tich)
# g.legend(title=r'$\xi_{\gamma}$', fontsize=7)
# fig.savefig(directory + '/Figures/Tikhonov_weight_comparison.png')
# fig.savefig(directory + '/Figures/Tikhonov_weight_comparison.pdf')
# plt.show(block=False)
# # Plot 2: fix regularization decay and alpha, vary epsilon decay
#
# fig, ax = plt.subplots(2,1, figsize=(4 * 1, 2.2 * 1), layout='constrained', sharex='col')
# g = sns.lineplot(data=df.loc[(df['Method']==TICH) & (df[r'$\alpha$']==alpha_to_plot) & (df['Reg. decay']==weight_reg_to_plot)],\
#                  drawstyle='steps-pre', errorbar=None, \
#                  estimator='mean', x='Iteration', palette='bright',
#                  y='Residual', hue='Approx. decay', style='Approx. decay', legend=False, linewidth = 2.0, ax=ax[0] )
# ax[0].grid(True)
# ax[0].set_yscale('log')
# ax[0].set_xscale('log')
# # ax[0].set_title("Compare ")
# ax[0].set_ylabel("Residual", fontsize = 9)
# ax[0].set_xlabel('Iteration', fontsize = 9)
# ax[0].set_xlim(1, N_iterations * 10 )
# ax[0].set_ylim(ax[0].get_ylim()[0]/2, ax[0].get_ylim()[1]*2 )
# # g.legend(labels = parameters_labels_tich)
# # g.legend(title=r'$\xi_{\varepsilon}$')
# g = sns.lineplot(data=df.loc[(df['Method']==TICH) & (df[r'$\alpha$']==alpha_to_plot) & (df['Reg. decay']==weight_reg_to_plot)], drawstyle='steps-pre', errorbar=None, \
#                  estimator='mean', x='Iteration', palette='bright',
#                  y='Rel. advantage', hue='Approx. decay', style='Approx. decay', linewidth = 2.0, ax=ax[1])
# ax[1].grid(True)
# ax[1].set_xscale('log')
# # ax[1,0].set_yscale('log')
# ax[1].set_ylabel(r'$\phi - \phi^{\mathrm{FBF}} (\%)$', fontsize = 9)
# ax[1].set_xlabel('Iteration', fontsize = 9)
# ax[1].set_xlim(1, N_iterations * 10 )
# ax[1].set_ylim(-1000, 0)
# ax[1].set_yticks(np.arange(-1000, 0 + 1, 500))
# # ax[1,0].set_ylim(ax[1,0].get_ylim()[0]/2, ax[1,0].get_ylim()[1]*2 )
# # g.legend(labels = parameters_labels_tich)
# legend_entries = []
# for eps in set(df.loc[(df['Method']==TICH) & (df[r'$\alpha$']==alpha_to_plot) & (df['Reg. decay']==weight_reg_to_plot)]['Approx. decay'].tolist()):
#     legend_entries.append(str(eps * weight_reg_to_plot))
# # g.legend(legend_entries, title=r'$\xi_{\varepsilon}$', fontsize=9)
# g.legend(title=r'$\xi_{\varepsilon}/\xi_{\gamma}$', fontsize=7)
# fig.savefig(directory + '/Figures/Tikhonov_epsilon_comparison.png')
# fig.savefig(directory + '/Figures/Tikhonov_epsilon_comparison.pdf')
#
# # Plot 3: fix weight decay and approx decay, vary alpha
# fig, ax = plt.subplots(2,1, figsize=(4 * 1, 2.2 * 1), layout='constrained', sharex='col')
# g = sns.lineplot(data=df.loc[(df['Method']==TICH) & (df['Approx. decay']==approx_reg_to_plot) & (df['Reg. decay']==weight_reg_to_plot)],\
#                  drawstyle='steps-pre', errorbar=None, \
#                  estimator='mean', x='Iteration', palette='bright',
#                  y='Residual', hue=r'$\alpha$', style=r'$\alpha$', legend=False, linewidth = 2.0, ax=ax[0] )
# ax[0].grid(True)
# ax[0].set_yscale('log')
# ax[0].set_xscale('log')
# # ax[0].set_title("Tichonov")
# ax[0].set_ylabel("Residual", fontsize = 9)
# ax[0].set_xlabel('Iteration', fontsize = 9)
# ax[0].set_xlim(1, N_iterations * 10 )
# ax[0].set_ylim(ax[0].get_ylim()[0]/2, ax[0].get_ylim()[1]*2 )
# # g.legend(labels = parameters_labels_tich)
# g = sns.lineplot(data=df.loc[(df['Method']==TICH) & (df['Approx. decay']==approx_reg_to_plot) & (df['Reg. decay']==weight_reg_to_plot)], drawstyle='steps-pre', errorbar=None, \
#                  estimator='mean', x='Iteration', palette='bright',
#                  y='Rel. advantage', hue=r'$\alpha$', style=r'$\alpha$', linewidth = 2.0, ax=ax[1])
# ax[1].grid(True)
# ax[1].set_xscale('log')
# # ax[1,0].set_yscale('log')
# ax[1].set_ylabel(r'$\phi - \phi^{\mathrm{FBF}} (\%)$', fontsize = 9)
# ax[1].set_xlabel('Iteration', fontsize = 9)
# ax[1].set_xlim(1, N_iterations * 10 )
# ax[1].set_ylim(-1000, 0)
# ax[1].set_yticks(np.arange(-1000, 0 + 1, 500))
# # ax[1,0].set_ylim(ax[1,0].get_ylim()[0]/2, ax[1,0].get_ylim()[1]*2 )
# # g.legend(labels = parameters_labels_tich)
# g.legend(title=r'$\alpha$', fontsize=7)
# plt.show(block=False)
# fig.savefig(directory + '/Figures/Tikhonov_alpha_comparison.png')
# fig.savefig(directory + '/Figures/Tikhonov_alpha_comparison.pdf')


("Done")