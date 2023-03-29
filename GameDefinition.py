import networkx
import torch
import numpy as np
import networkx as nx
from cmath import inf
from operators import backwardStep

torch.set_default_dtype(torch.float64)

class AggregativePartialInfo:
    # Define distributed aggregative game where each agent has the same number of opt. variables
    # Parameters:
    # Q \in R^N*n_x*n_x,  where Q[i,:,:] is the matrix that define the (quadratic) local cost
    # q \in R^N*n_x,  where q[i,:] is the affine part of the local cost
    # C \in R^N*n_s*n_x,  where C[i,:,:] is the matrix that define the local contribution to the aggregation
    # The aggregative variable is sigma = \sum (1/N) C_i x_i
    # D \in R^N*n_s*n_x,  where D[i,:,:] is the matrix that define the influence of the aggregation to the agent
    # A_shared \in R^N*n_m*n_x,  where A_shared[i,:,:] defines the local contribution to the shared eq. constraints
    # b_shared \in R^N*n_x,  where b_shared[i,:] is the affine part of the shared eq. constraints
    #### WARNING: D_iC_i should be symmetric!!
    # The game is in the form:
    # \sum .5 x_i' Q_i x_i + q_i'x_i + (1/N)(D_i x_i)'Cx
    # s.t. \sum_i A_shared_i x_i = \sum_i b_shared_i
    def __init__(self, N, communication_graph, Q, q, C, D, A_loc, b_loc, A_shared, b_shared, test=False):
        if test:
            N, n_opt_var, Q, c, Q_sel, c_sel, A_shared, b_shared, \
                A_eq_loc, A_ineq_loc, b_eq_loc, b_ineq_loc, communication_graph = self.setToTestGameSetup()
        self.N_agents = N
        self.n_opt_variables = Q.size(1)
        self.n_agg_variables = C.size(1)
        # Local constraints
        self.A_eq_loc = A_loc
        self.b_eq_loc = b_loc
        self.n_loc_eq_constr = self.A_eq_loc.size(1)
        # Shared constraints
        self.A_eq_shared= A_shared
        self.b_eq_shared = b_shared
        self.n_shared_eq_constr = self.A_eq_shared.size(1)
        # Define the (nonlinear) game mapping as a torch custom activation function
        self.F = self.GameMapping(Q, q, C, D)
        self.J = self.GameCost(Q, q, C, D)
        # Define the consensus operator
        # self.K = self.Consensus(communication_graph, self.n_shared_eq_constr)
        # Define the adjacency operator
        self.W = self.Adjacency(communication_graph)
        # Define the operator which computes the locally-estimated aggregation
        self.S = self.Aggregation(C)

    class GameCost(torch.nn.Module):
        def __init__(self, Q, q, C, D):
            super().__init__()
            self.Q = Q
            self.q = q
            self.C = C
            self.D = D
            self.N = Q.size(0)

        def forward(self, x):
            N = self.N
            agg = torch.sum(torch.bmm(self.C, x), dim=0).unsqueeze(0).repeat(N,1,1)
            cost = torch.bmm(x.transpose(1,2), torch.bmm(self.Q, x) + self.q) + (1/N)*torch.bmm(torch.transpose(torch.bmm(self.D,x),1,2), agg)
            return cost

    class GameMapping(torch.nn.Module):
        def __init__(self, Q, q, C, D):
            super().__init__()
            self.Q = Q
            self.q = q
            self.C = C
            self.D = D
            self.N = Q.size(0)
            self.n_x = Q.size(1)

        def forward(self, x, agg=None):
            # Optional argument agg allows to provide the estimated aggregation (Partial information)
            N = self.N
            if not agg is None:
                agg = torch.sum(torch.bmm(self.C, x), dim=0).unsqueeze(0).repeat(N,1,1)
            # F = Qx + q + (1/N)*(D_i'Cx + C_i'*D_i*x_i)
            pgrad = torch.bmm(self.Q, x) + self.q + (1 / N) * (
                        torch.bmm(torch.transpose(self.D, 1, 2), agg) + torch.bmm(torch.transpose(self.C,1,2), torch.bmm(self.D, x)))
            return pgrad

        def get_strMon_Lip_constants(self):
            # Return strong monotonicity and Lipschitz constant
            # Define the matrix that defines the pseudogradient mapping
            # F = Mx +m, where M = diag(Q_i) + diag(C_i'D_i) + col(D_i'C)
            N = self.Q.size(0)
            n_x = self.Q.size(2)
            diagonal_elements = self.Q + (1/N)*torch.bmm(torch.transpose(self.C,1,2), self.D)
            diagonal_elements_list = [diagonal_elements[i,:,:] for i in range(N)]
            Q_mat = torch.block_diag(*diagonal_elements_list)
            for i in range(N):
                for j in range(N):
                    Q_mat[i*n_x:(i+1)*n_x, j*n_x:(j+1)*n_x] = Q_mat[i*n_x:(i+1)*n_x, j*n_x:(j+1)*n_x] + \
                                                              torch.matmul(torch.transpose(self.D[i,:,:],0,1), self.C[j,:,:])
            U,S,V = torch.linalg.svd(Q_mat)
            return torch.min(S).item(), torch.max(S).item()

    class Consensus(torch.nn.Module):
        def __init__(self, communication_graph, N_dual_variables):
            super().__init__()
            # Convert Laplacian matrix to sparse tensor
            L = networkx.laplacian_matrix(communication_graph).tocoo()
            values = L.data
            rows = L.row
            cols = L.col
            indices = np.vstack((rows, cols))
            L = L.tocsr()
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            L_torch = torch.zeros(L.shape[0],L.shape[1], 1, 1)
            for i in rows:
                for j in cols:
                    L_torch[i,j,0,0] = L[i,j]
            # TODO: understand why sparse does not work
            # self.L = L_torch.to_sparse_coo()
            self.L = L_torch

        def forward(self, x):
            n_x = x.size(1)
            L_expanded = torch.kron(torch.eye(n_x).unsqueeze(0).unsqueeze(0), self.L)
            return torch.sum(torch.matmul(L_expanded, x), dim=1) # This applies the laplacian matrix to each of the dual variables

    class Adjacency(torch.nn.Module):
        def __init__(self, communication_graph):
            super().__init__()
            # Convert Laplacian matrix to sparse tensor
            W = networkx.adjacency_matrix(communication_graph).tocoo()
            values = W.data
            rows = W.row
            cols = W.col
            indices = np.vstack((rows, cols))
            W = W.tocsr()
            N=W.shape[0]
            W_torch = torch.zeros(N,N,1,1)
            for i in rows:
                for j in cols:
                    W_torch[i,j,0,0] = W[i,j]
            # TODO: understand why sparse does not work
            # self.L = L_torch.to_sparse_coo()
            self.W = W_torch

        def forward(self, x):
            n_x = x.size(1)
            W_expanded = torch.kron(torch.eye(n_x).unsqueeze(0).unsqueeze(0), self.W)
            return torch.sum(torch.matmul(W_expanded, x), dim=1) # This applies the adjacency matrix to each of the dual variables

    class Aggregation(torch.nn.Module):
        def __init__(self, C):
            super().__init__()
            self.C = C

        def forward(self, x):
            return torch.bmm(self.C,x)

    def setToTestGameSetup(self):
        raise NotImplementedError("[GameAggregativePartInfo:setToTestGameSetup] Test game not implemented")

    def get_strMon_Lip_constants_eq_constraints(self):
        N=self.N_agents
        ist_of_A_i = [self.A_eq_shared[i, :, :] for i in range(N)]
        list_of_A_i = [self.A_eq_shared[i, :, :] for i in range(N)]
        A = torch.column_stack(list_of_A_i)
        A_square = torch.matmul(A, torch.transpose(A, 0, 1))
        mu_A = torch.min(torch.linalg.eigvals(A_square).real)
        L_A = torch.sqrt(torch.max(torch.linalg.eigvals(A_square).real))
        return mu_A, L_A