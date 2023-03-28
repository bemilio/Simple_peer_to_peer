import numpy as np
import networkx as nx
import torch
import pickle
import GNE_part_info import primal_dual
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
    N_iter=100000
    N_it_per_residual_computation = 10
    N_agents = 10
    n_neighbors = 4 # for simplicity, each agent has the same number of neighbours. This is only used to create the communication graph (but i's not needed otherwise)
    N_random_tests = 1

    # Cost parameters
    c_mg = 1
    c_pr = 10
    c_tr = 1
    T = 24
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
            # pFB-Tichonov #TODO: fix!
            x_store_tich = torch.zeros(N_random_tests, game.N_agents, game.n_opt_variables)
            dual_store_tich = torch.zeros(N_random_tests, game.N_agents, game.n_shared_ineq_constr)
            aux_store_tich = torch.zeros(N_random_tests, game.N_agents, game.n_shared_ineq_constr)
            residual_store_tich = torch.zeros(N_random_tests, (N_iter // N_it_per_residual_computation))
            local_cost_store_tich = torch.zeros(N_random_tests, game.N_agents, (N_iter // N_it_per_residual_computation))
            sel_func_store_tich = torch.zeros(N_random_tests, (N_iter // N_it_per_residual_computation))

        #######################################
        #          GNE seeking                #
        #######################################
        # alg. initialization
        alg = primal_dual(game)
        alg.set_stepsize_using_Lip_const(safety_margin=.5)
        index_storage = 0
        avg_time_per_it_tich = 0
        for k in range(N_iter):
            if k % N_it_per_residual_computation == 0:
                # Save performance metrics
                x, d, a, r, c, s  = alg.get_state()
                residual_store_tich[test, index_parameter_set, index_storage] = r
                sel_func_store_tich[test, index_parameter_set, index_storage] = s
                print("Tichonov: Iteration " + str(k) + " Residual: " + str(r.item()) + " Sel function: " +  str(s.item()) +  " Average time: " + str(avg_time_per_it_tich))
                logging.info("Tichonov: Iteration " + str(k) + " Residual: " + str(r.item()) + " Sel function: " +  str(s.item()) +" Average time: " + str(avg_time_per_it_tich))
                index_storage = index_storage + 1
            #  Algorithm run
            start_time = time.time()
            alg_tich.run_once()
            end_time = time.time()
            avg_time_per_it_tich = (avg_time_per_it_tich * k + (end_time - start_time)) / (k + 1)

        # Store final variables
        x, d, a, r, c, s = alg_tich.get_state()
        x_store_tich[test, index_parameter_set, :, :] = x.flatten(1)
        dual_store_tich[test, index_parameter_set, :, :] = d.flatten(1)
        aux_store_tich[test, index_parameter_set, :, :] = a.flatten(1)
        local_cost_store_tich[test, index_parameter_set, :, :] = c.flatten(1)

    print("Saving results...")
    logging.info("Saving results...")
    f = open('saved_test_result_'+ str(job_id) + ".pkl", 'wb')
    pickle.dump([ x_store_tich, x_store_hsdm, x_store_std,
                 dual_store_tich, dual_store_hsdm, dual_store_std,
                 aux_store_tich, aux_store_hsdm, aux_store_std,
                 residual_store_tich, residual_store_hsdm, residual_store_std,
                 sel_func_store_tich, sel_func_store_hsdm, sel_func_store_std,
                 parameters_to_test_tich, parameters_to_test_hsdm], f)
    f.close()
    print("Saved")
    logging.info("Saved, job done")


