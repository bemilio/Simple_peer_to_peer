import numpy as np
import networkx as nx
import torch
import pickle
from GNE_part_info import primal_dual
from GameDefinition import AggregativePartialInfo
from SimpleP2PSetup import SimpleP2PSetup
import matplotlib.pyplot as plt
import time
import logging
import sys
import copy
import math

def gaussian(x, alpha, r):
    return 1. / (math.sqrt(alpha ** math.pi)) * np.exp(-alpha * np.power((x - r), 2.))

def generate_load_profile(N,T, variance):
    loads = torch.zeros(N,T,1)
    for i in range(N):
        peak_time = min(max(0.1*np.random.randn(), -1),1)
        steepness = 1/(max(0.3*np.random.randn() +1,.1))
        x = np.linspace(-1, 1, num=T)
        nominal_load = gaussian(x, steepness, peak_time)
        loads[i,:,0] = torch.from_numpy(nominal_load + variance*np.random.randn(T))
    return loads
def generate_gen_profile(N,T, variance):
    gen_profile = torch.zeros(N,T,1)
    nominal_profile = torch.matmul(torch.from_numpy(np.random.rand(N,1)), torch.ones(1,T) )
    gen_profile[:,:,0] = nominal_profile + torch.from_numpy(variance*np.random.randn(N,T))
    return gen_profile

if __name__ == '__main__':
    logging.basicConfig(filename='log.txt', filemode='w',level=logging.DEBUG)
    use_test_game = False  # trigger 2-players sample zero-sum monotone game
    if use_test_game:
        print("WARNING: test game will be used.")
        logging.info("WARNING: test game will be used.")
    if len(sys.argv) < 2:
        seed = 1
        job_id=0
    else:
        seed=int(sys.argv[1])
        job_id = int(sys.argv[2])
    print("Random seed set to  " + str(seed))
    logging.info("Random seed set to  " + str(seed))
    np.random.seed(seed)
    N_it_per_residual_computation = 10
    N_agents = 2
    n_neighbors = 1 # for simplicity, each agent has the same number of neighbours. This is only used to create the communication graph (but i's not needed otherwise)
    N_random_tests = 1

    # parameters
    c_mg = 10
    c_pr = 10
    c_tr = 1
    T = 24*4
    c_regul = 0.1
    N_iter = 10000
    N_iter_per_timestep = 100

    # Create load/gen profiles
    x_pr_setpoint = torch.ones(N_agents, T,1)
    loads = 5*torch.ones(N_agents, T,1)
    # x_pr_setpoint = generate_gen_profile(N_agents, T, 0)
    # loads = generate_load_profile(N_agents,T,0.01)



    ##########################################
    #  Define alg. parameters to test        #
    ##########################################

    for test in range(N_random_tests):
        ##########################################
        #        Test case creation              #
        ##########################################
        is_connected = False
        comm_graph = nx.random_regular_graph(n_neighbors, N_agents)
        while not nx.is_connected(comm_graph):
            n_neighbors = n_neighbors+1
            comm_graph = nx.random_regular_graph(n_neighbors, N_agents)
        # add self loops
        for i in comm_graph.nodes:
            comm_graph.add_edge(i,i)
        # Make graph stochastic WARNING: THIS IS ALSO DOUBLY STOCHASTIC ONLY BECAUSE WE ARE USING A REGULAR GRAPH
        comm_graph=nx.stochastic_graph(comm_graph.to_directed()).to_undirected()
        game_params = SimpleP2PSetup(N_agents, n_neighbors, comm_graph, c_mg, c_pr, c_tr, c_regul, T, x_pr_setpoint, loads)
        print("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
        logging.info("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
        ##########################################
        #             Game inizialization        #
        ##########################################
        game = AggregativePartialInfo(N_agents, comm_graph, game_params.Q, game_params.q, game_params.C, game_params.D,\
                                      game_params.A_eq_local_const, game_params.b_eq_local_const, \
                                      game_params.A_eq_shared_const, game_params.b_eq_shared_const)
        x_0 = torch.from_numpy(np.random.rand(game.N_agents, game.n_opt_variables, 1))  # torch.zeros(game.N_agents, game.n_opt_variables, 1)
        if test == 0:
            print("The game has " + str(game.N_agents) + " agents; " + str(game.n_opt_variables) + " opt. variables per agent; " \
                  + " local eq. constraints; " + str(game.n_shared_eq_constr) + " shared eq. constraints" )
            logging.info("The game has " + str(game.N_agents) + " agents; " + str(game.n_opt_variables) + " opt. variables per agent; " \
                  + str(game.A_eq_loc.size()[1]) + " local eq. constraints; " + str(game.n_shared_eq_constr) + " shared eq. constraints" )
            ##########################################
            #   Variables storage inizialization     #
            ##########################################
            # pFB-Tichonov
            x_store = torch.zeros(N_random_tests, game.N_agents, game.n_opt_variables)
            dual_share_store = torch.zeros(N_random_tests, game.N_agents, game.n_shared_eq_constr)
            dual_loc_store = torch.zeros(N_random_tests, game.N_agents, game.n_loc_eq_constr)
            aux_store = torch.zeros(N_random_tests, game.N_agents, game.n_shared_eq_constr)
            res_est_store = torch.zeros(N_random_tests, game.N_agents, game.n_shared_eq_constr)
            sigma_est_store = torch.zeros(N_random_tests, game.N_agents, game.n_agg_variables)
            residual_store = torch.zeros(N_random_tests, (N_iter // N_it_per_residual_computation))
            local_constr_viol = torch.zeros(N_random_tests, 1)
            shared_const_viol = torch.zeros(N_random_tests, 1)

        #######################################
        #          GNE seeking                #
        #######################################
        # alg. initialization
        alg = primal_dual(game)
        # The theoretically-sound stepsize is too small!
        # alg.set_stepsize_using_Lip_const(safety_margin=.9)
        index_storage = 0
        avg_time_per_it = 0
        for k in range(N_iter):
            if k % N_it_per_residual_computation == 0:
                # Save performance metrics
                x, d, d_l, aux, agg, res_est, r, c, const_viol_sh, const_viol_loc, dist_ref  = alg.get_state()
                residual_store[test, index_storage] = r
                print("Iteration " + str(k) + " Residual: " + str(r.item()) + " Average time: " + str(avg_time_per_it))
                logging.info("Iteration " + str(k) + " Residual: " + str(r.item()) +" Average time: " + str(avg_time_per_it))
                index_storage = index_storage + 1
            #  Algorithm run
            start_time = time.time()
            alg.run_once()
            end_time = time.time()
            avg_time_per_it = (avg_time_per_it * k + (end_time - start_time)) / (k + 1)

        # Store final variables
        x, d, d_l, aux, agg, res_est, r, c, const_viol_sh, const_viol_loc, dist_ref = alg.get_state()
        x_store[test, :, :] = x.flatten(1)
        dual_share_store[test, :, :] = d.flatten(1)
        dual_loc_store[test,:,:] = d_l.flatten(1)
        aux_store[test, :, :] = aux.flatten(1)
        sigma_est_store[test,:,:] = agg.flatten(1)
        res_est_store[test,:,:] = res_est.flatten(1)
        local_constr_viol[test] = const_viol_loc
        shared_const_viol[test] = const_viol_sh

        ############################
        #        time-var. test    #
        ############################

        for t in range(T):
            print("Initializing time-step" + str(t) + " out of " + str(T))
            logging.info("Initializing time-step" + str(t) + " out of " + str(T))
            game_params = SimpleP2PSetup(N_agents, n_neighbors, comm_graph, c_mg, c_pr, c_tr, c_regul, 1, x_pr_setpoint[:,t].unsqueeze(1),
                                         loads[:,t].unsqueeze(1))
            game_old = copy.deepcopy(game)
            game = AggregativePartialInfo(N_agents, comm_graph, game_params.Q, game_params.q, game_params.C,
                                          game_params.D, \
                                          game_params.A_eq_local_const, game_params.b_eq_local_const, \
                                          game_params.A_eq_shared_const, game_params.b_eq_shared_const)
            # initialize the variables to be stored over the simulation period
            if test==0 and t==0:
                n = game.n_opt_variables  # For simplicity every agent has the same n. of variables
                s = game.n_agg_variables
                m = game.n_shared_eq_constr
                m_loc = game.n_loc_eq_constr
                x_tvar = torch.zeros(N_random_tests, T, N_agents, n)
                agg_tvar = torch.zeros(N_random_tests, T, N_agents, s)
                res_tvar = torch.zeros(N_random_tests, T, N_agents, m)
                dual_tvar = torch.zeros(N_random_tests, T, N_agents, m)
                aux_tvar = torch.zeros(N_random_tests, T, N_agents, m)
                dual_loc_tvar = torch.zeros(N_random_tests, T, N_agents, m_loc)
                # Performance metrics
                shared_const_viol_tvar = torch.zeros(N_random_tests,T)
                loc_const_viol_tvar = torch.zeros(N_random_tests, T)
                distance_from_optimal_tvar = torch.zeros(N_random_tests, T)
            if t==0:
                x_tvar[test, 0, :, :] = torch.from_numpy(np.random.rand(N_agents, n))
                alg = primal_dual(game, x_0=x_tvar[test, 0, :, :].unsqueeze(2))
            else:
                # alg. re-initialization
                x_init = x_tvar[test, t-1, :, :].unsqueeze(2)
                agg_init = agg_tvar[test,t-1,:,:].unsqueeze(2) - game_old.S(x_init) + game.S(x_init)
                res_init = res_tvar[test,t-1,:,:].unsqueeze(2) - game_old.b_eq_shared + game.b_eq_shared
                dual_init = dual_tvar[test,t-1,:,:].unsqueeze(2)
                aux_init = aux_tvar[test,t-1,:,:].unsqueeze(2)
                dual_loc_init = dual_loc_tvar[test,t-1,:,:].unsqueeze(2)
                alg = primal_dual(game, x_0=x_init, agg_0=agg_init, res_0=res_init, dual_0=dual_init, aux_0=aux_init, dual_loc_0=dual_loc_init)
            for k in range(N_iter_per_timestep):
                #  Algorithm run
                alg.run_once()

            # Compute P-distance with respect to pre-computed GNE
            x_ref = x_store[test, :, t*n:(t+1)*n].unsqueeze(2)
            d_ref = dual_share_store[test, :, t*m:(t+1)*m].unsqueeze(2)
            d_loc_ref = dual_loc_store[test, :, t*m_loc:(t+1)*m_loc].unsqueeze(2)
            d_ref_avg = torch.mean(d_ref, dim=0)
            x_ref = torch.reshape(x_ref, (x_ref.size(0) * x_ref.size(1), 1))
            d_loc_ref = torch.reshape(d_loc_ref, (d_loc_ref.size(0) * d_loc_ref.size(1), 1))
            omega_ref = torch.row_stack((x_ref, d_ref_avg, d_loc_ref))
            x, d, d_l, aux, agg, res_est, r, c, const_viol_sh, const_viol_loc, dist_ref = alg.get_state(omega_ref)
            # store computed decision variables (THESE ARE ALSO USED FOR THE RE-INITIALIZATION)
            x_tvar[test, t, : ,:] =x.flatten(1)
            agg_tvar[test, t, : ,:] =agg.flatten(1)
            res_tvar[test, t, : ,:] =res_est.flatten(1)
            dual_tvar[test, t, : ,:] =d.flatten(1)
            aux_tvar[test, t, : ,:] = aux.flatten(1)
            dual_loc_tvar[test, t, : ,:] =d_l.flatten(1)
            # Store performance variables
            loc_const_viol_tvar[test, t] = const_viol_loc
            shared_const_viol_tvar[test, t] = const_viol_sh
            distance_from_optimal_tvar[test,t] = dist_ref
            print("Timestep " + str(t) + " Distance from ref.: " + str(dist_ref.item()), " Constr. violation: " + str(const_viol_sh.item() + const_viol_loc.item()))
            logging.info("Timestep " + str(t) + " Distance from ref.: " + str(dist_ref.item()))

    print("Saving results...")
    logging.info("Saving results...")
    f = open('saved_test_result_'+ str(job_id) + ".pkl", 'wb')
    pickle.dump([ x_store, residual_store, dual_share_store, dual_loc_store,
                  local_constr_viol, shared_const_viol,
                  loc_const_viol_tvar, shared_const_viol_tvar,
                  distance_from_optimal_tvar, game_params.edge_to_index ], f)
    f.close()
    print("Saved")
    logging.info("Saved, job done")


