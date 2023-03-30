import numpy as np
import torch
from operators.backwardStep import BackwardStep

class primal_dual: # For partial information aggregative games with only shared equality constr.
    def __init__(self, game, x_0=None, agg_0=None, res_0=None, dual_0=None, aux_0=None, dual_loc_0=None, stepsize=0.01):
        self.game = game
        # self.P = self.set_stepsize_using_Lip_const(safety_margin)
        self.stepsize = stepsize
        self.N = game.N_agents
        self.P, self.nu = self.compute_P_matrix()
        n = game.n_opt_variables # For simplicity every agent has the same n. of variables
        s = game.n_agg_variables
        m = game.n_shared_eq_constr
        m_loc = game.n_loc_eq_constr
        if x_0 is not None:
            self.x = x_0
        else:
            self.x = torch.zeros(self.N, n, 1)
        if agg_0 is not None:
            self.agg = agg_0
        else:
            self.agg = self.game.S(self.x)
        if res_0 is not None:
            self.res = res_0
        else:
            self.res = torch.bmm(self.game.A_eq_shared, self.x) - self.game.b_eq_shared
        if aux_0 is not None:
            self.aux = aux_0
        else:
            self.aux = torch.zeros(self.N,m, 1)
        if dual_0 is not None:
            self.dual = dual_0
        else:
            self.dual = self.aux
        if dual_loc_0 is not None:
            self.dual_loc = dual_loc_0
        else:
            self.dual_loc = torch.zeros(self.N,m_loc, 1)
        # These are used to store the previous iteration value (needed for residual computation)
        self.x_last = self.x
        self.dual_last = self.dual
        self.dual_loc_last = self.dual_loc
        self.aux_last = self.aux
        self.res_last = self.res
        self.agg_last = self.agg

    def run_once(self):
        x = self.x
        agg = self.agg
        res = self.res
        dual = self.dual
        dual_loc = self.dual_loc
        aux = self.aux
        A_i = self.game.A_eq_shared
        b_i = self.game.b_eq_shared
        A_i_loc = self.game.A_eq_loc
        b_i_loc = self.game.b_eq_loc
        F = self.game.F(x,agg)

        # run updates
        x_new = x - self.stepsize * (F + torch.bmm(torch.transpose(A_i, 1, 2), self.dual) + torch.bmm(torch.transpose(A_i_loc, 1, 2), self.dual_loc))
        dual_loc_new = dual_loc + self.stepsize * (torch.bmm(A_i_loc, x) - b_i_loc)
        aux_new = aux + self.stepsize * self.N * res
        # the function game.W applies the incidence matrix, the function game.S computes the aggregation
        agg_new = self.game.W(agg) + self.game.S(x_new) - self.game.S(x)
        res_new = self.game.W(res) + torch.bmm(A_i,x_new-x)
        dual_new = self.game.W(dual) + aux_new - aux

        self.x = x_new
        self.aux = aux_new
        self.agg = agg_new
        self.res = res_new
        self.dual = dual_new
        self.dual_loc = dual_loc_new

        self.x_last = x
        self.dual_last = dual
        self.dual_loc_last = dual_loc
        self.aux_last = aux
        self.res_last = res
        self.agg_last = agg

    def get_state(self, ref_point=None):
        residual,  constr_viol_sh, constr_viol_loc = self.compute_residual()
        cost = self.game.J(self.x)
        if ref_point is not None:
            dist_ref = self.compute_distance_from_ref(ref_point)
        else:
            dist_ref=None
        return self.x, self.dual, self.dual_loc, self.aux, self.agg, self.res, residual, cost, constr_viol_sh, constr_viol_loc, dist_ref

    def compute_distance_from_ref(self, ref_x):
        x = self.x
        # d_avg = torch.mean(self.dual, dim=0)
        # d_loc = self.dual_loc
        # x = torch.reshape(x, (x.size(0) * x.size(1), 1))
        # d_loc = torch.reshape(d_loc, (d_loc.size(0) * d_loc.size(1), 1))
        # omega_1 = torch.row_stack((x, d_avg, d_loc))
        # dist_ref = torch.matmul(torch.matmul(torch.transpose(omega_1-ref_point,0,1), torch.from_numpy(self.P)), omega_1-ref_point)
        dist_ref = torch.norm(x-ref_x)
        return dist_ref

    def compute_residual(self):
        # As the game is strongly monotone, the convergence is checked by x_{t+1} - x_t.
        # A_i = self.game.A_eq_shared
        # b_i = self.game.b_eq_shared
        # x = self.x
        # x_res, status = self.game.F(x) - torch.matmul(torch.transpose(A_i, 1, 2), self.dual)
        # d_res = torch.bmm(A_i, self.x) - b_i
        # residual = np.sqrt( ((x_res).norm())**2 + ((d_res).norm())**2 )

        P = self.P
        x = self.x
        d_avg = torch.mean(self.dual, dim=0)
        A_sh = self.game.A_eq_shared
        b_sh = torch.sum(self.game.b_eq_shared, dim=0)
        A_i_loc = self.game.A_eq_loc
        b_i_loc = self.game.b_eq_loc
        # reshape everything in a column vector
        res_x = self.game.F(x) + torch.matmul(torch.transpose(A_sh, 1,2), d_avg) + torch.bmm(torch.transpose(A_i_loc, 1, 2), self.dual_loc)
        res_d_sh = torch.sum(torch.bmm(A_sh, x), dim=0) - torch.sum(b_sh, dim=0)
        res_d_loc = torch.bmm(A_i_loc, x)- b_i_loc
        res_x = torch.reshape(res_x, (res_x.size(0) * res_x.size(1), 1))
        res_d_loc = torch.reshape(res_d_loc, (res_d_loc.size(0) * res_d_loc.size(1), 1) )
        res_avg_track = torch.norm(self.dual - d_avg*torch.ones(self.dual.size()))**2
        res_res_track = torch.norm(self.res - torch.mean(self.res,dim=0) * torch.ones(self.res.size()))**2
        res_agg_track = torch.norm(self.agg - torch.mean(self.agg, dim=0) * torch.ones(self.agg.size()))**2

        omega_1_res = torch.row_stack((res_x, res_d_sh, res_d_loc))
        residual = .5*torch.matmul(torch.matmul( torch.transpose(omega_1_res, 0,1), torch.from_numpy(P)), omega_1_res) \
                   + res_avg_track + res_res_track + res_agg_track
        constr_viol_sh = torch.norm(res_d_sh )
        constr_viol_loc = torch.sqrt(torch.norm(res_d_loc )**2 + \
                          torch.norm(torch.minimum(torch.bmm(self.game.A_sel_positive_vars,x), torch.zeros(x.size()) ))**2)
        return residual, constr_viol_sh, constr_viol_loc



    def compute_P_matrix(self):
        mu_F, L_F = self.game.F.get_strMon_Lip_constants()
        n = self.game.n_opt_variables
        N = self.game.N_agents
        m_sh = self.game.n_shared_eq_constr
        m_loc = self.game.n_loc_eq_constr
        list_of_A_sh_i = [self.game.A_eq_shared[i, :, :] for i in range(N)]
        list_of_A_loc_i = [self.game.A_eq_loc[i,:,:] for i in range(N)]
        A = torch.row_stack( (torch.column_stack(list_of_A_sh_i), torch.block_diag(*list_of_A_loc_i)) )
        mu_A, L_A = self.game.get_strMon_Lip_constants_eq_constraints()
        nu = .5 * 4 * mu_F * mu_A / (L_F * L_F * L_A * L_A + 4 * mu_A * L_A * L_A)
        P = np.block([[np.eye(n * N), np.array(nu * torch.transpose(A, 0, 1).numpy())],
                      [np.array(nu * A.numpy()), np.eye(m_sh + m_loc*N)]])
        return P, nu

    def set_stepsize_using_Lip_const(self, safety_margin=0.5):

        mu_F, L_F = self.game.F.get_strMon_Lip_constants()
        P = self.P
        nu = self.nu
        mu_A, L_A = self.game.get_strMon_Lip_constants_eq_constraints()
        M = np.matrix([ [ mu_F-nu*L_A*L_A,    -nu*L_A*L_F/2 ],
                        [ -nu * L_A * L_F/2,   nu*mu_A      ]])
        # compute str. monotonicity and lipschitz constant in P-norm
        mu_KKT_P = np.min(np.linalg.eigvals(M).real)/np.min(np.linalg.eigvals(P).real)
        L_KKT_P_square = (L_F+L_A)*np.max(np.linalg.eigvals(P).real)/np.min(np.linalg.eigvals(P).real)
        self.stepsize = safety_margin * 2 *mu_KKT_P/L_KKT_P_square
        return P