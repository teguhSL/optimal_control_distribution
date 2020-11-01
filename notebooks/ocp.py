#TODO: modify to accept different Qt and Rt

import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import inv
from ocp_sys import *
from scipy.stats import multivariate_normal as mvn

    
###LQR###
    
class finiteLQR():
    def __init__(self, sys):
        self.sys = sys
        self.A = sys.A
        self.B = sys.B
        self.Dx = self.sys.Dx
        self.Du = self.sys.Du
            
        self.x_ref = np.zeros(self.Dx)
        
    def reset_AB(self, A, B):
        self.A = A
        self.B = B
        self.sys.reset_AB(A,B)
        
    def set_cost(self, Q, R, Qf = None):
        if Q.ndim == 2:
            self.Q = np.array([Q]*(self.T+1))
            self.R = np.array([R]*(self.T+1))
            if Qf is not None:
                self.Q[-1] = Qf
        elif Q.ndim == 3:
            self.Q = Q
            self.R = R
        else:
            print('Number of dimensions must be either 2 or 3')
            #raise()    
            
    def set_timestep(self,T):
        self.T = T
        
    def set_xref(self, x_ref):
        self.x_ref = x_ref
            
    def calc_cost(self):
        cost = 0
        for i in range(self.T):
            cost += (self.xs[i]-self.x_ref).T.dot(self.Q[i]).dot(self.xs[i]-self.x_ref) + self.us[i].T.dot(self.R[i]).dot(self.us[i])
        cost += (self.xs[self.T]-self.x_ref).T.dot(self.Q[self.T]).dot(self.xs[self.T]-self.x_ref)
        self.cost = cost
        return cost
        
    def compute_Pt(self):
        P = np.zeros((self.T+1, self.Dx, self.Dx))
        P[-1] = self.Q[-1]
        for i in np.arange(self.T,0,-1):
            P[i-1] = self.Q[i-1] + self.A.T.dot(P[i]).dot(self.A)  - self.A.T.dot(P[i]).dot(self.B).dot(inv(self.R[i-1]+self.B.T.dot(P[i]).dot(self.B))).dot(self.B.T).dot(P[i]).dot(self.A)
        self.P = P
        return P
    
    def solve(self, method = 'DP'):
        #Method: can be 'DP' (Dynamic programming) or 'LS' (Least Square)
        if method == 'DP':
            self.compute_Pt()
            x = self.sys.x0.copy()
            xs = [x]
            us = []
            Ks = []
            for i in range(self.T):
                K = inv(self.R[i] + self.B.T.dot(self.P[i+1]).dot(self.B)).dot(self.B.T).dot(self.P[i+1]).dot(self.A)
                u = -K.dot(x-self.x_ref)
                x = self.sys.step(x, u)
                xs += [x]
                us += [u]
                Ks += [K]
            self.xs = np.array(xs)
            self.us = np.array(us)
            self.Ks = np.array(Ks)
            
            return self.xs, self.us
        elif method == 'LS':
            Qs = np.zeros(((self.T+1)*self.Dx,(self.T+1)*self.Dx))
            Rs = np.zeros(((self.T+1)*self.Du,(self.T+1)*self.Du))

            for i in range(self.T+1):
                Qs[self.Dx*i:self.Dx*(i+1),self.Dx*i:self.Dx*(i+1)] = self.Q[i]
                Rs[self.Du*i:self.Du*(i+1),self.Du*i:self.Du*(i+1)] = self.R[i]
            
            Sx = np.zeros((self.Dx*(self.T+1),self.Dx))
            Su = np.zeros((self.Dx*(self.T+1),self.Du*(self.T+1)))

            #### Calculate Sx and Su 
            i = 0
            Sx[self.Dx*i:self.Dx*(i+1), :] = np.eye(self.Dx)
            for i in range(1, self.T+1):
                Sx[self.Dx*i:self.Dx*(i+1), :] =  Sx[self.Dx*(i-1):self.Dx*(i), :].dot(self.A)

            for i in range(self.T+1):
                Su[self.Dx*(i+1):,(self.Du*i):(self.Du*(i+1))] = Sx[:self.Dx*(self.T+1-(i+1))].dot(self.B)

            #### Calculate X and U 
            Sigma_u_inv = (Su.T.dot(Qs.dot(Su)) + Rs)
            us = np.linalg.solve(Sigma_u_inv, Su.T.dot(Qs.dot(Sx.dot(-self.sys.x0))))
            xs = Sx.dot(self.sys.x0) + Su.dot(us)
            xs = xs.reshape(self.T+1,self.Dx)

            self.xs = xs
            self.us = us[:,None]

            self.Qs = Qs
            self.Rs = Rs
            self.Sx = Sx
            self.Su = Su
            self.Sigma_u_inv = Sigma_u_inv
            return self.xs, self.us
        else:
            print('Method is unknown!')
            return None, None
    
    
    def sample(self, n = 1, recreate_dist = True):
        us = self.us.flatten()
        xs = self.xs.flatten()
        if recreate_dist:
            self.dist = mvn(mean = us, cov = inv(self.Sigma_u_inv))
            
        sample_xs = []
        sample_us = []
        
        for i in range(n):
            sample_u = self.dist.rvs()
            sample_x = self.Sx.dot(self.sys.x0) + self.Su.dot(sample_u)
            
            sample_xs += [sample_x.reshape(-1,self.Dx)]
            sample_us += [sample_u.reshape(-1,self.Du)]
        return np.array(sample_xs), np.array(sample_us)
    
    
from scipy import linalg as la
class infiniteLQR(finiteLQR):                  
    def compute_Pt(self):
        self.P = la.solve_discrete_are(self.A,self.B,self.Q,self.R)
        return self.P
    
    def solve(self):
        self.compute_Pt()
        x = self.sys.x0.copy()
        xs = [x]
        us = []
        for i in range(self.T):
            K = inv(self.R + self.B.T.dot(self.P).dot(self.B)).dot(self.B.T).dot(self.P).dot(self.A)
            u = -K.dot(x-self.x_ref)
            x = self.A.dot(x) + self.B.dot(u)
            xs += [x]
            us += [u]
        self.xs = np.array(xs)
        self.us = np.array(us)
        return self.xs, self.us
    
    
    
###LQT###
#follow the algorithm in : http://karimpor.profcms.um.ac.ir/imagesm/354/stories/papers/karimpour_kiomarcy.pdf
#Optimal Tracking Control for Linear Discrete-time Systems Using Reinforcement Learning. Bahare Kiumarsi-Khomartash, Frank L. Lewis, Fellow, IEEE, Mohammad-Bagher Naghibi-Sistani,
#and Ali Karimpour

class finiteLQT(finiteLQR):
    def set_ref(self, x_ref):
        self.x_ref = x_ref
            
    def calc_cost(self):
        cost = 0
        for i in range(self.T):
            cost += (self.xs[i]-self.x_ref[i]).T.dot(self.Q[i]).dot(self.xs[i]-self.x_ref[i]) + self.us[i].T.dot(self.R[i]).dot(self.us[i])
        cost += (self.xs[i]-self.x_ref[i]).T.dot(self.Q[self.T]).dot(self.xs[i]-self.x_ref[i])
        self.cost = cost
        return cost
        
    def compute_Pt(self):
        P = np.zeros((self.T+1, self.Dx, self.Dx))
        P[-1] = self.Q[-1]
        for i in np.arange(self.T,0,-1):
            P[i-1] = self.Q[i-1] + self.A.T.dot(P[i]).dot(self.A)  - self.A.T.dot(P[i]).dot(self.B).dot(inv(self.R[i-1]+self.B.T.dot(P[i]).dot(self.B))).dot(self.B.T).dot(P[i]).dot(self.A)
        self.P = P
        return P
    
    def compute_gain(self):
        Ks = []
        Kvs = []
        for i in range(self.T):
            K = inv(self.B.T.dot(self.P[i+1]).dot(self.B) + self.R[i]).dot(self.B.T).dot(self.P[i+1]).dot(self.A)
            Kv = inv(self.B.T.dot(self.P[i+1]).dot(self.B) + self.R[i]).dot(self.B.T)
            Ks += [K]
            Kvs += [Kv]
        self.Ks = np.array(Ks)
        self.Kvs = np.array(Kvs)
    
    def compute_vref(self):
        vs = np.zeros((self.T+1, self.Dx))
        vs[-1] = self.Q[-1].dot(self.x_ref[-1])
        for i in np.arange(self.T,0,-1):
            vs[i-1] = (self.A-self.B.dot(self.Ks[i-1])).T.dot(vs[i]) + self.Q[i-1].dot(self.x_ref[i-1])
        self.vs = vs
        
    def solve(self, method = 'DP'):
        if method == 'DP':
            self.compute_Pt()
            self.compute_gain()
            self.compute_vref()

            x = self.sys.x0.copy()
            xs = [x]
            us = []
            for i in range(self.T):
                u = -self.Ks[i].dot(x) + self.Kvs[i].dot(self.vs[i+1])
                x = self.sys.step(x, u)#self.A.dot(x) + self.B.dot(u)
                xs += [x]
                us += [u]
            self.xs = np.array(xs)
            self.us = np.array(us)
            return self.xs, self.us
        elif method == 'LS':
            Qs = np.zeros(((self.T+1)*self.Dx,(self.T+1)*self.Dx))
            Rs = np.zeros(((self.T+1)*self.Du,(self.T+1)*self.Du))

            for i in range(self.T+1):
                Qs[self.Dx*i:self.Dx*(i+1),self.Dx*i:self.Dx*(i+1)] = self.Q[i]
                Rs[self.Du*i:self.Du*(i+1),self.Du*i:self.Du*(i+1)] = self.R[i]

            #### Calculate Sx and Su 
            Sx = np.zeros((self.Dx*(self.T+1),self.Dx))
            Su = np.zeros((self.Dx*(self.T+1),self.Du*(self.T+1)))

            i = 0
            Sx[self.Dx*i:self.Dx*i+2, :] = np.eye(self.Dx)
            for i in range(1, self.T+1):
                Sx[self.Dx*i:self.Dx*(i+1), :] =  Sx[self.Dx*(i-1):self.Dx*(i), :].dot(self.A)

            for i in range(self.T+1):
                Su[self.Dx*(i+1):,(self.Du*i):(self.Du*(i+1))] = Sx[:self.Dx*(self.T+1-(i+1))].dot(self.B)

            #### Calculate X and U 
            Sigma_u_inv = (Su.T.dot(Qs.dot(Su)) + Rs)
            us = np.linalg.solve(Sigma_u_inv, Su.T.dot(Qs.dot(self.x_ref.flatten() + Sx.dot(-self.sys.x0))))
            xs = Sx.dot(self.sys.x0) + Su.dot(us)
            xs = xs.reshape(self.T+1,self.Dx)

            self.xs = xs
            self.us = us[:,None]

            self.Qs = Qs
            self.Rs = Rs
            self.Sx = Sx
            self.Su = Su
            self.Sigma_u_inv = Sigma_u_inv

            return self.xs, self.us
        else:
            print('Method is unknown!')
            return None, None

        

class ILQR_Standard():
    '''
    ILQR Standard: uses the standard quadratic cost function Q, R, and Qf
    This class is kept only for educational purpose, as it is simpler than the one 
    using cost model
    '''
    def __init__(self, sys, mu = 1e-6):
        self.sys, self.Dx, self.Du = sys, sys.Dx, sys.Du
        self.mu = mu
        
    def set_timestep(self,T):
        self.T = T
        self.allocate_data()
        
    def set_reg(self,mu):
        self.mu = mu
        
    def set_ref(self, x_refs):
        self.x_refs = x_refs.copy()
        
    def allocate_data(self):
        self.Lx  = np.zeros((self.T+1, self.Dx)) 
        self.Lu  = np.zeros((self.T+1,   self.Du))
        self.Lxx = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Luu = np.zeros((self.T+1,   self.Du, self.Du))
        self.Fx  = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Fu  = np.zeros((self.T+1, self.Dx, self.Du))
        self.Vx  = np.zeros((self.T+1, self.Dx))
        self.Vxx = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Qx  = np.zeros((self.T,   self.Dx))
        self.Qu  = np.zeros((self.T,   self.Du))
        self.Qux = np.zeros((self.T,   self.Du, self.Dx))
        self.Qxx = np.zeros((self.T,   self.Dx, self.Dx))
        self.Quu = np.zeros((self.T,   self.Du, self.Du))
        self.k = np.zeros((self.T, self.Du))
        self.K = np.zeros((self.T, self.Du, self.Dx))
        
        self.xs = np.zeros((self.T+1, self.Dx))
        self.us = np.zeros((self.T+1, self.Du))
        self.x_refs = np.zeros((self.T+1, self.Dx))

    def set_cost(self, Q, R, Qf = None):
        if Q.ndim == 2:
            self.Q = np.array([Q]*(self.T+1))
            self.R = np.array([R]*(self.T+1)) #note: the last R is only created for convenience, u_T does not affect anything and will be zero
            if Qf is not None:
                self.Q[-1] = Qf
        elif Q.ndim == 3:
            self.Q = Q
            self.R = R
        else:
            print('Number of dimensions must be either 2 or 3')
            #raise()    
                
    def set_init_state(self,x0):
        self.x0 = x0.copy()
        
    def set_state(self, xs, us):
        self.xs = xs.copy()
        self.us = us.copy()
        
    def calc_diff(self):
        for i in range(self.T+1):
            self.Lx[i] = self.Q[i].dot(self.xs[i]- self.x_refs[i])
            self.Lxx[i] = self.Q[i]
            self.Luu[i] = self.R[i]
            self.Lu[i] = self.R[i].dot(self.us[i])
            self.Fx[i], self.Fu[i] = self.sys.compute_matrices(self.xs[i], self.us[i])
            
    def calc_cost(self, xs, us):
        running_cost_state = 0
        running_cost_control = 0
        cost = 0
        #for i in range(self.T):
        #    cost += (xs[i]- self.x_refs[i]).T.dot(self.Q[i]).dot(xs[i]- self.x_refs[i]) + us[i].T.dot(self.R[i]).dot(us[i])
        #cost += (xs[self.T]- self.x_refs[i]).T.dot(self.Q[self.T]).dot(xs[self.T]- self.x_refs[i])
        for i in range(self.T):
            running_cost_state += (xs[i]- self.x_refs[i]).T.dot(self.Q[i]).dot(xs[i]- self.x_refs[i])
            running_cost_control += us[i].T.dot(self.R[i]).dot(us[i])
        terminal_cost_state = (xs[self.T]- self.x_refs[i]).T.dot(self.Q[self.T]).dot(xs[self.T]- self.x_refs[i])
        self.cost = running_cost_state + running_cost_control + terminal_cost_state
        self.running_cost_state = running_cost_state
        self.running_cost_control = running_cost_control
        self.terminal_cost_state = terminal_cost_state
        return self.cost
    
    def calc_dcost(dxs, dus):
        #need to call 'compute_du_LS' first
        return 0.5*dxs.T.dot(self.Qs).dot(dxs) + 0.5*dus.T.dot(self.Rs).dot(dus) + self.Lxs.dot(dxs) + self.Lus.dot(dus)
    
    def forward_pass(self, max_iter = 20):
        print('Starting line searches ...')
        cost0 = self.calc_cost(self.xs, self.us)
        print(cost0)
        alpha = 1.
        fac = 0.5
        cost = 5*cost0
        del_us = []
        n_iter = 0
        while cost > cost0 and n_iter < max_iter  :
            xs_new = []
            us_new = []
            x = self.x0.copy()
            xs_new += [x]
            for i in range(self.T):
                del_u = alpha*self.k[i] + self.K[i].dot(x-self.xs[i])
                u = self.us[i] + del_u
                x = self.sys.step(x,u)
                xs_new += [x]
                us_new += [u]
                del_us += [del_u]
            
            us_new += [np.zeros(self.Du)]  #add the last control as 0, for convenience
            cost = self.calc_cost(xs_new,us_new)
            print(alpha,cost)
            alpha *= fac
            n_iter += 1
        print('Completing line search ... \n')
            
        self.xs, self.us = np.array(xs_new), np.array(us_new)
        self.del_us = np.array(del_us)
    
    def backward_pass(self):
        self.Vx[self.T] = self.Lx[self.T]
        self.Vxx[self.T] = self.Lxx[self.T]
        for i in np.arange(self.T-1, -1,-1):
            self.Qx[i] = self.Lx[i]   + self.Fx[i].T.dot(self.Vx[i+1])
            self.Qu[i] = self.Lu[i]   + self.Fu[i].T.dot(self.Vx[i+1])
            self.Qxx[i] = self.Lxx[i] + self.Fx[i].T.dot(self.Vxx[i+1]).dot(self.Fx[i])
            self.Quu[i] = self.Luu[i] + self.Fu[i].T.dot(self.Vxx[i+1]).dot(self.Fu[i]) + self.mu*np.eye(self.Du)
            self.Qux[i] = self.Fu[i].T.dot(self.Vxx[i+1]).dot(self.Fx[i])
            Quuinv_i = inv(self.Quu[i])
            self.k[i] = -Quuinv_i.dot(self.Qu[i])
            self.K[i] = -Quuinv_i.dot(self.Qux[i])

            self.Vx[i] = self.Qx[i] - self.Qu[i].dot(Quuinv_i).dot(self.Qux[i])
            self.Vxx[i] = self.Qxx[i] - self.Qux[i].T.dot(Quuinv_i).dot(self.Qux[i])
            #ensure symmetrical Vxx
            self.Vxx[i] = 0.5*(self.Vxx[i] + self.Vxx[i].T)
    
    def solve(self, n_iter = 3):
        for i in range(n_iter):
            self.calc_diff()
            self.backward_pass()
            self.forward_pass()
            
    def compute_du_LS(self):
        self.Qs = np.zeros(((self.T+1)*self.Dx,(self.T+1)*self.Dx))
        self.Rs = np.zeros(((self.T+1)*self.Du,(self.T+1)*self.Du))
        
        for i in range(self.T+1):
            self.Qs[self.Dx*i:self.Dx*(i+1),self.Dx*i:self.Dx*(i+1)] = self.Lxx[i]
            self.Rs[self.Du*i:self.Du*(i+1),self.Du*i:self.Du*(i+1)] = self.Luu[i]

        self.Sx = np.zeros((self.Dx*(self.T+1),self.Dx))
        self.Su = np.zeros((self.Dx*(self.T+1),self.Du*(self.T+1)))

        #### Calculate Sx and Su 
        i = 0
        self.Sx[self.Dx*i:self.Dx*(i+1), :] = np.eye(self.Dx)
        for i in range(1, self.T+1):
            self.Sx[self.Dx*i:self.Dx*(i+1), :] =  self.Sx[self.Dx*(i-1):self.Dx*(i), :].dot(self.Fx[i-1])

        for i in range(1,self.T+1):
            self.Su[self.Dx*i:self.Dx*(i+1), self.Du*(i-1): self.Du*(i)] = self.Fu[i-1]
            self.Su[self.Dx*i:self.Dx*(i+1), :self.Du*(i-1)] = self.Fx[i-1].dot(self.Su[self.Dx*(i-1):self.Dx*(i), :self.Du*(i-1)])

        self.Lxs = self.Lx.flatten()
        self.Lus = self.Lu.flatten()

        #### Calculate X and U 
        self.Sigma_u_inv = (self.Su.T.dot(self.Qs.dot(self.Su)) + self.Rs) + self.mu*np.eye(self.Rs.shape[0])
        self.del_us_ls = -np.linalg.solve(self.Sigma_u_inv, self.Su.T.dot(self.Qs.dot(self.Sx.dot(-np.zeros(self.Dx)))) + self.Lxs.dot(self.Su) + self.Lus )
        self.del_xs_ls = self.Sx.dot(np.zeros(self.Dx)) + self.Su.dot(self.del_us_ls)
        return self.del_xs_ls, self.del_us_ls, self.Sigma_u_inv
    
    def sample_du(self, n = 1, recreate_dist = True, allow_singular = True):
        print('sampling')
        mean_del_us = self.del_us_ls
        self.Sigma_u = inv(self.Sigma_u_inv)
        self.Sigma_x = self.Su.dot(self.Sigma_u).dot(self.Su.T)
        if recreate_dist:
            self.dist = mvn(mean = mean_del_us, cov = self.Sigma_u, allow_singular = allow_singular)
            
        dx0 = np.zeros(self.Dx)
        Sx0 = self.Sx.dot(dx0)
        sample_dxs = []
        sample_dus = self.dist.rvs(n)
        for i in range(n):
            sample_del_us = sample_dus[i].flatten()
            sample_del_xs = Sx0 + self.Su.dot(sample_del_us)
            sample_dxs += [sample_del_xs]
            
        return  np.array(sample_dxs), np.array(sample_dus)
#missing: better line search and regularization


class ILQR():
    def __init__(self, sys, mu = 1e-6):
        self.sys, self.Dx, self.Du = sys, sys.Dx, sys.Du
        self.mu = mu
        
    def set_timestep(self,T):
        self.T = T
        self.allocate_data()
        
    def set_reg(self,mu):
        self.mu = mu
        
    def set_ref(self, x_refs):
        self.x_refs = x_refs.copy()
        
    def allocate_data(self):
        self.Lx  = np.zeros((self.T+1, self.Dx)) 
        self.Lu  = np.zeros((self.T+1,   self.Du))
        self.Lxx = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Luu = np.zeros((self.T+1,   self.Du, self.Du))
        self.Fx  = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Fu  = np.zeros((self.T+1, self.Dx, self.Du))
        self.Vx  = np.zeros((self.T+1, self.Dx))
        self.Vxx = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Qx  = np.zeros((self.T,   self.Dx))
        self.Qu  = np.zeros((self.T,   self.Du))
        self.Qux = np.zeros((self.T,   self.Du, self.Dx))
        self.Qxx = np.zeros((self.T,   self.Dx, self.Dx))
        self.Quu = np.zeros((self.T,   self.Du, self.Du))
        self.k = np.zeros((self.T, self.Du))
        self.K = np.zeros((self.T, self.Du, self.Dx))
        
        self.xs = np.zeros((self.T+1, self.Dx))
        self.us = np.zeros((self.T+1, self.Du))
        self.x_refs = np.zeros((self.T+1, self.Dx))

    def set_cost(self, costs):
        self.costs = costs
                
    def set_init_state(self,x0):
        self.x0 = x0.copy()
        
    def set_state(self, xs, us):
        self.xs = xs.copy()
        self.us = us.copy()
        
    def calc_diff(self):
        for i in range(self.T+1):
            self.costs[i].calcDiff(self.xs[i], self.us[i])
            self.Lx[i]  = self.costs[i].Lx
            self.Lxx[i] = self.costs[i].Lxx
            self.Lu[i]  = self.costs[i].Lu
            self.Luu[i] = self.costs[i].Luu
            self.Fx[i], self.Fu[i] = self.sys.compute_matrices(self.xs[i], self.us[i])
            
    def calc_cost(self, xs, us):
        self.cost = np.sum([self.costs[i].calc(xs[i], us[i]) for i in range(self.T+1)])
        return self.cost
    
    def calc_dcost(dxs, dus):
        #need to call 'compute_du_LS' first
        return 0.5*dxs.T.dot(self.Qs).dot(dxs) + 0.5*dus.T.dot(self.Rs).dot(dus) + self.Lxs.dot(dxs) + self.Lus.dot(dus)
    
    def forward_pass(self, max_iter = 10):
        cost0 = self.calc_cost(self.xs, self.us)
        print(cost0)
        alpha = 1.
        fac = 0.8
        cost = 5*cost0
        
        n_iter = 0
        while cost > cost0 and n_iter < max_iter  :
            xs_new = []
            us_new = []
            x = self.x0.copy()
            xs_new += [x]
            for i in range(self.T):
                u = self.us[i] + alpha*self.k[i] + self.K[i].dot(x-self.xs[i])
                x = self.sys.step(x,u)
                xs_new += [x]
                us_new += [u]
            
            us_new += [np.zeros(self.Du)]  #add the last control as 0, for convenience
            cost = self.calc_cost(xs_new,us_new)
            print(alpha,cost)
            alpha *= fac
            n_iter += 1
        self.xs, self.us = np.array(xs_new), np.array(us_new)
    
    
    def backward_pass(self):
        self.Vx[self.T] = self.Lx[self.T]
        self.Vxx[self.T] = self.Lxx[self.T]
        for i in np.arange(self.T-1, -1,-1):
            self.Qx[i] = self.Lx[i]   + self.Fx[i].T.dot(self.Vx[i+1])
            self.Qu[i] = self.Lu[i]   + self.Fu[i].T.dot(self.Vx[i+1])
            self.Qxx[i] = self.Lxx[i] + self.Fx[i].T.dot(self.Vxx[i+1]).dot(self.Fx[i])
            self.Quu[i] = self.Luu[i] + self.Fu[i].T.dot(self.Vxx[i+1]).dot(self.Fu[i]) + self.mu*np.eye(self.Du)
            self.Qux[i] = self.Fu[i].T.dot(self.Vxx[i+1]).dot(self.Fx[i])
            Quuinv = inv(self.Quu[i])
            self.k[i] = -Quuinv.dot(self.Qu[i])
            self.K[i] = -Quuinv.dot(self.Qux[i])

            self.Vx[i] = self.Qx[i] - self.Qu[i].dot(Quuinv).dot(self.Qux[i])
            self.Vxx[i] = self.Qxx[i] - self.Qux[i].T.dot(Quuinv).dot(self.Qux[i])
            #ensure symmetrical Vxx
            self.Vxx[i] = 0.5*(self.Vxx[i] + self.Vxx[i].T)

    
    
    def solve(self, n_iter = 3):
        for i in range(n_iter):
            self.calc_diff()
            self.backward_pass()
            self.forward_pass()
            
    def compute_du_LS(self):
        self.Qs = np.zeros(((self.T+1)*self.Dx,(self.T+1)*self.Dx))
        self.Rs = np.zeros(((self.T+1)*self.Du,(self.T+1)*self.Du))
        
        for i in range(self.T+1):
            self.Qs[self.Dx*i:self.Dx*(i+1),self.Dx*i:self.Dx*(i+1)] = self.Lxx[i]
            self.Rs[self.Du*i:self.Du*(i+1),self.Du*i:self.Du*(i+1)] = self.Luu[i]

        self.Sx = np.zeros((self.Dx*(self.T+1),self.Dx))
        self.Su = np.zeros((self.Dx*(self.T+1),self.Du*(self.T+1)))

        #### Calculate Sx and Su 
        i = 0
        self.Sx[self.Dx*i:self.Dx*(i+1), :] = np.eye(self.Dx)
        for i in range(1, self.T+1):
            self.Sx[self.Dx*i:self.Dx*(i+1), :] =  self.Sx[self.Dx*(i-1):self.Dx*(i), :].dot(self.Fx[i-1])

        for i in range(1,self.T+1):
            self.Su[self.Dx*i:self.Dx*(i+1), self.Du*(i-1): self.Du*(i)] = self.Fu[i-1]
            self.Su[self.Dx*i:self.Dx*(i+1), :self.Du*(i-1)] = self.Fx[i-1].dot(self.Su[self.Dx*(i-1):self.Dx*(i), :self.Du*(i-1)])

        self.Lxs = self.Lx.flatten()
        self.Lus = self.Lu.flatten()

        #### Calculate X and U 
        self.Sigma_u_inv = (self.Su.T.dot(self.Qs.dot(self.Su)) + self.Rs)
        self.del_us_ls = -np.linalg.solve(self.Sigma_u_inv, self.Su.T.dot(self.Qs.dot(self.Sx.dot(-np.zeros(self.Dx)))) + self.Lxs.dot(self.Su) + self.Lus )
        self.del_xs_ls = self.Sx.dot(np.zeros(self.Dx)) + self.Su.dot(self.del_us_ls)
        return self.del_xs_ls, self.del_us_ls, self.Sigma_u_inv
    

    def sample_du(self, n = 1, recreate_dist = True):
        mean_del_us = self.del_us_ls
        self.Sigma_u = inv(self.Sigma_u_inv)
        self.Sigma_x = self.Su.dot(self.Sigma_u).dot(self.Su.T)
        if recreate_dist:
            self.dist = mvn(mean = mean_del_us, cov = inv(self.Sigma_u_inv))
            
        dx0 = np.zeros(self.Dx)
        Sx0 = self.Sx.dot(dx0)
        sample_dxs = []
        sample_dus = self.dist.rvs(n)
        
        for i in range(n):
            sample_del_us = sample_dus[i].flatten()
            sample_del_xs = Sx0 + self.Su.dot(sample_del_us)
            sample_dxs += [sample_del_xs]
            
        return np.array(sample_dxs), np.array(sample_dus)

def get_ilqr_from_ddp(ddp, ilqr):
    T = ddp.problem.T
    ilqr.set_timestep(T)
    ilqr.xs = np.array(ddp.xs)
    ilqr.us = np.concatenate([ddp.us, np.zeros((1, len(ddp.us[-1])))])

    datas = ddp.problem.runningDatas
    for i in range(T):
        ilqr.Fx[i] = datas[i].Fx
        ilqr.Fu[i] = datas[i].Fu
        ilqr.Lx[i] = datas[i].Lx.flatten()
        ilqr.Lu[i] = datas[i].Lu.flatten()
        ilqr.Lxx[i] = datas[i].Lxx
        ilqr.Luu[i] = datas[i].Luu

    data = ddp.problem.terminalData
    ilqr.Fx[T] = data.Fx
    ilqr.Fu[T] = data.Fu
    ilqr.Lx[T] = data.Lx.flatten()
    ilqr.Lu[T] = data.Lu.flatten()
    ilqr.Lxx[T] = data.Lxx
    ilqr.Luu[T] = data.Luu
    ilqr.backward_pass()
    return ilqr