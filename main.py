import numpy as np
import networkx as nx
import torch
import pickle
from GNE_part_info import primal_dual
from GameDefinition import AggregativePartialInfo
from SimpleP2PSetup import SimpleP2PSetup

import time
import logging
import sys

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
    N_iter=10000
    N_it_per_residual_computation = 10
    N_agents = 2
    n_neighbors = 1 # for simplicity, each agent has the same number of neighbours. This is only used to create the communication graph (but i's not needed otherwise)
    N_random_tests = 1

    # Cost parameters
    c_mg = 10
    c_pr = 10
    c_tr = 1
    T = 1
    c_regul = 0.1

    # Create load/gen profiles
    x_pr_setpoint = torch.ones(N_agents, T,1)
    loads = 5*torch.ones(N_agents, T,1)

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
            omega_variation_store = torch.zeros(N_random_tests, (N_iter // N_it_per_residual_computation))
            local_cost_store = torch.zeros(N_random_tests, game.N_agents, (N_iter // N_it_per_residual_computation))

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
                x, d, d_l, aux, agg, res_est, r, c  = alg.get_state()
                omega_variation_store[test, index_storage] = r
                local_cost_store[test, :,index_storage] = c.flatten(0)
                print("Iteration " + str(k) + " Residual: " + str(r.item()) + " Average time: " + str(avg_time_per_it))
                logging.info("Iteration " + str(k) + " Residual: " + str(r.item()) +" Average time: " + str(avg_time_per_it))
                index_storage = index_storage + 1
            #  Algorithm run
            start_time = time.time()
            alg.run_once()
            end_time = time.time()
            avg_time_per_it = (avg_time_per_it * k + (end_time - start_time)) / (k + 1)

        # Store final variables
        x, d, d_l, aux, agg, res_est, r, c  = alg.get_state()
        x_store[test, :, :] = x.flatten(1)
        dual_share_store[test, :, :] = d.flatten(1)
        dual_loc_store[test,:,:] = d_l.flatten(1)
        aux_store[test, :, :] = aux.flatten(1)
        sigma_est_store[test,:,:] = agg.flatten(1)
        res_est_store[test,:,:] = res_est.flatten(1)

    print("Saving results...")
    logging.info("Saving results...")
    f = open('saved_test_result_'+ str(job_id) + ".pkl", 'wb')
    pickle.dump([ x_store, dual_share_store, dual_loc_store,
                 aux_store, sigma_est_store, res_est_store,
                 omega_variation_store, local_cost_store, game_params.edge_to_index], f)
    f.close()
    print("Saved")
    logging.info("Saved, job done")


