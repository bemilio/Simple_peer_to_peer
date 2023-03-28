import numpy as np
import torch
from operators.backwardStep import BackwardStep

class primal_dual: # For partial information aggregative games with only shared equality constr.
    def __init__(self, game, x_0=None, agg_0=None, res_0=None, dual_0=None, aux_0=None, dual_loc_0=None,
                 stepsize=0.001):
        self.game = game
        self.stepsize = stepsize
        self.N = game.N_agents
        n = game.n_opt_variables # For simplicity every agent has the same n. of variables
        s = game.n_agg_variables
        m = game.n_shared_eq_constr
        m_loc = game.n_loc_eq_constr
        if x_0 is not None:
            self.x = x_0
        else:
            self.x = torch.zeros(self.N, n, 1)
        if agg_0:
            self.agg = agg_0
        else:
            self.agg = torch.zeros(self.N, s, 1)
        if res_0:
            self.res = res_0
        else:
            self.res = torch.zeros(self.N, m, 1)
        if dual_0:
            self.dual = dual_0
        else:
            self.dual = torch.zeros(self.N,m, 1)
        if aux_0:
            self.aux = aux_0
        else:
            self.aux = torch.zeros(self.N,m, 1)
        if dual_loc_0:
            self.dual_loc = dual_loc_0
        else:
            self.dual_loc = torch.zeros(self.N,m_loc, 1)

    def run_once(self):
        x = self.x
        agg = self.agg
        res = self.res
        dual = self.dual
        aux = self.aux
        A_i = self.game.A_eq_shared
        b_i = self.game.b_eq_shared
        A_i_loc = self.game.A_eq_loc
        b_i_loc = self.game.b_eq_loc
        F = self.game.F(x,agg)

        # run updates
        x_new = x - self.stepsize * (F + torch.bmm(torch.transpose(A_i, 1, 2), self.dual) +  torch.bmm(torch.transpose(A_i_loc, 1, 2), self.dual_loc))
        self.dual_loc = self.dual_loc + self.stepsize * (torch.bmm(A_i_loc, x) - b_i_loc)
        aux_new = aux + self.stepsize * self.N * res
        # the function game.W applies the incidence matrix, the function game.S computes the aggregation
        agg_new = self.game.W(agg) + self.game.S(x_new) - self.game.S(x)
        res_new = self.game.W(res) + torch.bmm(A_i,x_new) - torch.bmm(A_i,x)
        dual_new = self.game.W(dual) + aux_new - aux

        self.x = x_new
        self.aux = aux_new
        self.agg = agg_new
        self.res = res_new
        self.dual = dual_new

    def get_state(self):
        residual = self.compute_residual()
        cost = self.game.J(self.x)
        return self.x, self.dual, self.aux, residual, cost

    def compute_residual(self):
        # As the game is strongly monotone, the convergence is better checked by x_{t+1} - x_t instead of this residual.
        A_i = self.game.A_eq_shared
        b_i = self.game.b_eq_shared
        x = self.x
        x_res, status = self.game.F(x) - torch.matmul(torch.transpose(A_i, 1, 2), self.dual)
        d_res = torch.bmm(A_i, self.x) - b_i
        residual = np.sqrt( ((x_res).norm())**2 + ((d_res).norm())**2 )
        return residual

    def set_stepsize_using_Lip_const(self, safety_margin=0.5):
        mu_F, L_F = self.game.F.get_strMon_Lip_constants()
        n = self.game.n_opt_variables
        m = self.game.n_shared_eq_constr
        A_i = self.game.A_eq_shared
        A = torch.sum(A_i, dim=0)
        A_square = torch.matmul(A, torch.transpose(A, 0,1))
        mu_A = torch.min(torch.linalg.eig(A_square))
        L_A = torch.sqrt(torch.max(torch.linalg.eig(A_square)))
        nu = safety_margin * 4*mu_F*mu_A / (L_F*L_F*L_A*L_A + 4*mu_A*L_A)
        P = np.block( [[np.eye(n), nu*torch.transpose(A, 0,1).numpy() ],
                       [nu*A.numpy(), np.eye(m) ] ] )
        M = np.matrix([ [ mu_F-nu*L_A*L_A,    -nu*L_A*L_F/2 ],
                        [ -nu * L_A * L_F/2,   nu*mu_A      ]])
        # compute str. monotonicity and lipschitz constant in P-norm
        mu_KKT_P = np.min(np.linalg.eig(M))/np.min(np.linalg.eig(P))
        L_KKT_P_square = (L_F+L_A)*np.max(np.linalg.eig(P))/np.min(np.linalg.eig(P))
        self.stepsize = safety_margin * 2 *mu_KKT_P/L_KKT_P_square
