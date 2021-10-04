import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import inv
from utils import computePose, computeJacobian
class CostModelQuadratic():
    def __init__(self, sys, Q = None, R = None, x_ref = None, u_ref = None):
        self.sys = sys
        self.Dx, self.Du = sys.Dx, sys.Du
        self.Q, self.R = Q, R
        if Q is None: self.Q = np.zeros((self.Dx,self.Dx))
        if R is None: self.R = np.zeros((self.Du,self.Du))
        self.x_ref, self.u_ref = x_ref, u_ref
        if x_ref is None: self.x_ref = np.zeros(self.Dx)
        if u_ref is None: self.u_ref = np.zeros(self.Du)
            
    def set_ref(self, x_ref=None, u_ref=None):
        if x_ref is not None:
            self.x_ref = x_ref
        if u_ref is not None:
            self.u_ref = u_ref
    
    def calc(self, x, u):
        self.L = 0.5*(x-self.x_ref).T.dot(self.Q).dot(x-self.x_ref) + 0.5*(u-self.u_ref).T.dot(self.R).dot(u-self.u_ref)
        return self.L
    
    def calcDiff(self, x, u):
        self.Lx = self.Q.dot(x-self.x_ref)
        self.Lu = self.R.dot(u-self.u_ref)
        self.Lxx = self.Q.copy()
        self.Luu = self.R.copy()
        self.Lxu = np.zeros((self.Dx, self.Du))
        
class CostModelSum():
    def __init__(self, sys, costs):
        self.sys = sys
        self.costs = costs
        self.Dx, self.Du = sys.Dx, sys.Du
    
    def calc(self, x, u):
        self.L = 0
        for i,cost in enumerate(self.costs):
            cost.calc(x, u)
            self.L += cost.L
        return self.L
    
    def calcDiff(self, x, u):
        self.Lx = np.zeros(self.Dx)
        self.Lu = np.zeros(self.Du)
        self.Lxx = np.zeros((self.Dx,self.Dx))
        self.Luu = np.zeros((self.Du,self.Du))
        self.Lxu = np.zeros((self.Dx,self.Du))
        for i,cost in enumerate(self.costs):
            cost.calcDiff(x, u)
            self.Lx += cost.Lx
            self.Lu += cost.Lu
            self.Lxx += cost.Lxx
            self.Luu += cost.Luu
            self.Lxu += cost.Lxu
            
class CostModelQuadraticTranslation():
    '''
    The quadratic cost model for the end effector, p = f(x)
    '''
    def __init__(self, sys, W, p_ref = None):
        self.sys = sys
        self.Dx, self.Du = sys.Dx, sys.Du
        self.W = W
        self.p_ref = p_ref
        if p_ref is None: self.p_ref = np.zeros(3)
            
    def set_ref(self, p_ref):
        self.p_ref = p_ref
        
    def calc(self, x, u):
        p,_ = self.sys.compute_ee(x)
        self.L = 0.5*(p-self.p_ref).T.dot(self.W).dot(p-self.p_ref) 
        return self.L
    
    def calcDiff(self, x, u):
        self.J   = self.sys.compute_Jacobian(x)
        p,_      = self.sys.compute_ee(x)
        self.Lx  = self.J.T.dot(self.W).dot(p-self.p_ref)
        self.Lx = np.concatenate([self.Lx, np.zeros(self.Dx/2)])
        self.Lu  = np.zeros(self.Du)
        self.Lxx = np.zeros((self.Dx, self.Dx))
        self.Lxx[:self.Dx/2, :self.Dx/2] = self.J.T.dot(self.W).dot(self.J)
        self.Luu = np.zeros((self.Du, self.Du))
        self.Lxu = np.zeros((self.Dx, self.Du))
        
class ActivationCollision():
    def __init__(self, nr, threshold=0.3):
        self.threshold = threshold
        self.nr = nr
    def calc(self,  r):
        self.d = np.linalg.norm(r)

        if self.d < self.threshold:
            self.a = 0.5*(self.d-self.threshold)**2
        else:
            self.a = 0
        return self.a

    def calcDiff(self,  r, recalc=True):
        if recalc:
            self.calc(r)
        
        if self.d < self.threshold:
            self.Ar = (self.d-self.threshold)*r[:,None]/self.d
            self.Arr = np.eye(self.nr)*(self.d-self.threshold)/self.d + self.threshold*np.outer(r,r.T)/(self.d**3)
        else:
            self.Ar = np.zeros((self.nr,1))
            self.Arr = np.zeros((self.nr,self.nr))
        return self.Ar, self.Arr


class SphereSphereCollisionCost():
    def __init__(self, activation=None, nu=None, r_body = 0., r_obs = 0., pos_obs = np.array([0,0,0]), w= 1.):
        self.activation = activation 
        self.r_body = r_body
        self.r_obs = r_obs
        self.pos_obs = pos_obs
        self.w = w
        
    def calc(self, x, u):              
        #calculate residual
        pos_body = x[:3]
        self.r = pos_body-self.pos_obs
        self.activation.calc(self.r)
        self.L = self.activation.a*self.w
        return self.L

    def calcDiff(self,  x, u, recalc=True):
        if recalc:
            self.calc( x, u)

        #Calculate the Jacobian at p1
        self.J = np.hstack([np.eye(3), np.zeros((3,3))])
        ###Compute the cost derivatives###
        self.activation.calcDiff(self.r, recalc)
        self.Rx = np.hstack([self.J, np.zeros((3, 6))])
        self.Lx = np.vstack([self.J.T.dot(self.activation.Ar), np.zeros((6, 1))])
        self.Lxx = np.vstack([
              np.hstack([self.J.T.dot(self.activation.Arr).dot(self.J),
                      np.zeros((6, 6))]),
           np.zeros((6, 12))
        ])*self.w
        self.Lx = self.Lx[:,0]*self.w
        self.Lu = np.zeros(4)
        self.Luu = np.zeros((4,4))
        self.Lxu = np.zeros((12, 4))
        return self.Lx, self.Lxx
    
class SphereCapsuleCollisionCost():
    def __init__(self, activation=None, nu=None, p1 = np.zeros(3), p2 = np.zeros(3),  w= 1.):
        self.activation = activation        
        self.p1 = p1
        self.p2 = p2
        self.length = np.linalg.norm(self.p2-self.p1)
        self.v = (p2-p1)/self.length
        self.w = w
        
    def calc(self, x, u):              
        #calculate residual
        p = x[:3]
        pt = p - self.p1 
        t = pt.dot(self.v)
        if t < 0:
            r = p - self.p1
        elif t > self.length:
            r = p - self.p2
        else:
            r = pt - t*self.v

        self.pt = pt
        self.t = t
        self.r = r
        self.activation.calc(self.r)
        self.L = self.activation.a*self.w
        return self.L

    def calcDiff(self,  x, u, recalc=True):
        if recalc:
            self.calc( x, u)

        #Calculate the Jacobian at p1
        self.J = np.hstack([np.eye(3), np.zeros((3,3))])
        ###Compute the cost derivatives###
        self.activation.calcDiff(self.r, recalc)
        self.Rx = np.hstack([self.J, np.zeros((3, 6))])
        self.Lx = np.vstack([self.J.T.dot(self.activation.Ar), np.zeros((6, 1))])
        self.Lxx = np.vstack([
              np.hstack([self.J.T.dot(self.activation.Arr).dot(self.J),
                      np.zeros((6, 6))]),
           np.zeros((6, 12))
        ])*self.w
        self.Lx = self.Lx[:,0]*self.w
        self.Lu = np.zeros(4)
        self.Luu = np.zeros((4,4))
        self.Lxu = np.zeros((12, 4))
        return self.Lx, self.Lxx
    
def num_diff(f, x, d_out = 1, inc = 0.001):
    J = np.zeros((d_out, len(x)))
    u = np.zeros(4)
    for i in range(len(x)):
        x_ip = x.copy()
        x_ip[i] += inc
        val_p = f(x_ip, u)
        x_im = x.copy()
        x_im[i] -= inc
        val_m = f(x_im, u)

        
        diff = (val_p - val_m)/(2*inc)
        J[:,i] = diff
    return J.T

def num_calc_diff(x, d_out = 1, inc = 0.001):
    J = np.zeros((d_out, len(x)))
    u = np.zeros(4)
    f = cost_collision.calcDiff
    for i in range(len(x)):
        x_ip = x.copy()
        x_ip[i] += inc
        val_p = f(x_ip, u)[0].flatten()
        x_im = x.copy()
        x_im[i] -= inc
        val_m = f(x_im, u)[0].flatten()
        
        diff = (val_p - val_m)/(2*inc)
        J[:,i] = diff
    return J


class RobotSphereCollisionCost():
    def __init__(self, activation=None, nu=None, rmodel = None, rdata = None, ee_id = 0, r_body = 0., r_obs = 0., pos_obs = np.array([0,0,0]), w= 1.):
        self.activation = activation 
        self.r_body = r_body
        self.r_obs = r_obs
        self.pos_obs = pos_obs
        self.w = w
        self.rmodel = rmodel
        self.rdata = rdata
        self.ee_id = ee_id
        
    def calc(self, x, u):              
        #calculate the position of the end-effector
        pos_body, ori = computePose(self.rmodel, self.rdata, self.ee_id, x[:7])
        self.r = pos_body-self.pos_obs
        self.activation.calc(self.r)
        self.L = self.activation.a*self.w
        return self.L

    def calcDiff(self,  x, u, recalc=True):
        if recalc:
            self.calc( x, u)

        #Calculate the Jacobian at p1
        #self.J = np.hstack([np.eye(3), np.zeros((3,3))])
        self.J = computeJacobian(self.rmodel, self.rdata, self.ee_id, x[:7])[:3]
        ###Compute the cost derivatives###
        self.activation.calcDiff(self.r, recalc)
        self.Rx = np.hstack([self.J, np.zeros((3, 7))])
        self.Lx = np.vstack([self.J.T.dot(self.activation.Ar), np.zeros((7, 1))])
        self.Lxx = np.vstack([
              np.hstack([self.J.T.dot(self.activation.Arr).dot(self.J),
                      np.zeros((7, 7))]),
           np.zeros((7, 14))
        ])*self.w
        self.Lx = self.Lx[:,0]*self.w
        self.Lu = np.zeros(7)
        self.Luu = np.zeros((7,7))
        self.Lxu = np.zeros((14, 7))
        return self.Lx, self.Lxx
    
class CostModelBound:
    """
    This cost is to keep within the joint limits
    """
    def __init__(self, sys, bounds, weight=1., margin = 1e-3): 
        self.bounds = bounds
        self.dof = bounds.shape[1]
        self.Dx, self.Du = sys.Dx, sys.Du
        self.identity = np.eye(self.dof)
        self.margin = margin
        self.weight = weight
        
    def calc(self, x, u):
        self.res = ((x - self.bounds[0]) * (x < self.bounds[0]) +  \
                    (x - self.bounds[1]) * ( x > self.bounds[1]))
        self.L = 0.5*self.weight*(self.res.dot(self.res))
        return
    
    def calcDiff(self, x, u, recalc = True):
        if recalc:
            self.calc(x,u)        
        stat = (x - self.margin < self.bounds[0]) + \
                (x + self.margin > self.bounds[1])
        self.J = stat*self.identity
        self.Lx = self.weight*self.J.dot(self.res)
        self.Lxx = self.weight*self.J.T.dot(self.J)
        self.Lu  = np.zeros(self.Du)
        self.Lxu = np.zeros((self.Dx, self.Du))
        self.Luu  = np.zeros((self.Du, self.Du))
        return  