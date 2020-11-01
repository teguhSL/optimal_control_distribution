import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
from casadi import mtimes, MX, sin, cos, vertcat, horzcat, sum1, cross, Function, jacobian
import casadi
from utils import *
import pinocchio
import crocoddyl

class LinearSystem():
    def __init__(self,A,B):
        self.A = A
        self.B = B
        self.Dx = A.shape[0]
        self.Du = B.shape[1]
        
    def reset_AB(self, A,B):
        self.A = A
        self.B = B
        
    def set_init_state(self,x0):
        self.x0 = x0
    
    def step(self, x, u):
        return self.A.dot(x) + self.B.dot(u)
    
    def rollout(self,us):
        x_cur = self.x0
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)

class Unicycle():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 3
        self.Du = 2
        
    def set_init_state(self,x0):
        self.x0 = x0

    def compute_matrices(self,x,u):
        A = np.eye(3)
        A[0,2] = -u[0]*np.sin(x[2])*self.dt
        A[1,2] = u[0]*np.cos(x[2])*self.dt
        
        B = np.zeros((3,2))
        B[0,0] = np.cos(x[2])*self.dt
        B[1,0] = np.sin(x[2])*self.dt
        B[2,1] = 1*self.dt
        self.A, self.B = A,B
        return A,B
        
    def step(self, x, u):
        #A,B = self.compute_matrices(x,u)
        x_next = np.zeros(3)
        
        x_next[0] = x[0] + u[0]*np.cos(x[2])*self.dt
        x_next[1] = x[1] + u[0]*np.sin(x[2])*self.dt
        x_next[2] = x[2] + u[1]*self.dt
        #pdb.set_trace()
        return x_next
        #return A.dot(x) + B.dot(u)
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
class SecondUnicycle():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 5
        self.Du = 2
        
    def set_init_state(self,x0):
        self.x0 = x0

    def compute_matrices(self,x,u):
        A = np.eye(self.Dx)
        A[0,2] = -x[3]*np.sin(x[2])*self.dt
        A[1,2] = x[3]*np.cos(x[2])*self.dt
        
        A[0,3] = np.cos(x[2])*self.dt
        A[1,3] = np.sin(x[2])*self.dt
        A[2,4] = 1*self.dt
        
        B = np.zeros((self.Dx,self.Du))
        B[3,0] = self.dt
        B[4,1] = self.dt
        
        self.A, self.B = A,B
        return A,B
        
    def step(self, x, u):
        x_next = np.zeros(self.Dx)
        
        x_next[0] = x[0] + x[3]*np.cos(x[2])*self.dt
        x_next[1] = x[1] + x[3]*np.sin(x[2])*self.dt
        x_next[2] = x[2] + x[4]*self.dt
        x_next[3] = x[3] + u[0]*self.dt
        x_next[4] = x[4] + u[1]*self.dt
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
class Pendulum():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 2
        self.Du = 1
        self.b = 1
        self.m = 1
        self.l = 1
        
    def set_init_state(self,x0):
        self.x0 = x0

    def compute_matrices(self,x,u):
        A = np.eye(self.Dx)
        B = np.zeros((self.Dx,self.Du))
        
        A[0,1] = self.dt
        A[1,0] = 0.5*9.8*self.dt*np.cos(x[0])/self.l
        A[1,1] = 1 - self.dt*self.b/(self.m*self.l**2)
        
        B[1,0] = self.dt/(self.m*self.l**2)
        
        self.A, self.B = A,B
        return A,B
        
    def step(self, x, u):
        x_next = np.zeros(self.Dx)
        x_next[0] = x[0] + x[1]*self.dt
        x_next[1] = (1-self.dt*self.b/(self.m*self.l**2))*x[1] + 0.5*9.8*self.dt*np.sin(x[0])/self.l + self.dt*u/(self.m*self.l**2) 
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
    def plot(self, x, color='k'):
        px = np.array([0, -self.l*np.sin(x[0])])
        py = np.array([0, self.l*np.cos(x[0])])
        line = plt.plot(px, py, marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
        xlim = [-2*self.l, 2*self.l]
        plt.axes().set_aspect('equal')
        plt.axis(xlim+xlim)
        return line

    def plot_traj(self, xs, dt = 0.1, filename = None):
        for i,x in enumerate(xs):
            clear_output(wait=True)
            self.plot(x)
            if filename is not None:
                plt.savefig('temp/fig'+str(i)+'.png')
            plt.show()
            time.sleep(dt)
    
    
class Bicopter():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 6
        self.Du = 2
        
        self.m = 2.5
        self.l = 1
        self.I = 1.2
        
    def set_init_state(self,x0):
        self.x0 = x0

    def compute_matrices(self,x,u):
        A = np.eye(self.Dx)
        A[:3, 3:] = np.eye(3)*self.dt
        A[3,2] = -self.dt*(u[0]+u[1])*np.cos(x[2])/self.m
        A[4,2] = -self.dt*(u[0]+u[1])*np.sin(x[2])/self.m
        
        B = np.zeros((self.Dx,self.Du))
        B[3,0] = -self.dt*np.sin(x[2])/self.m
        B[3,1] = B[3,0]
        
        B[4,0] = self.dt*np.cos(x[2])/self.m
        B[4,1] = B[4,0]
        
        B[5,0] = self.dt*self.l*0.5/self.I
        B[5,1] = -B[5,0]
        
        self.A, self.B = A,B
        return A,B
        
    def step(self, x, u):
        x_next = np.zeros(self.Dx)
        
        x_next[0] = x[0] + x[3]*self.dt
        x_next[1] = x[1] + x[4]*self.dt
        x_next[2] = x[2] + x[5]*self.dt
        
        x_next[3] = x[3] - (u[0]+u[1])*np.sin(x[2])*self.dt/self.m
        x_next[4] = x[4] + (u[0]+u[1])*np.cos(x[2])*self.dt/self.m - 9.8*self.dt
        x_next[5] = x[5] + (u[0]-u[1])*self.dt*self.l*0.5/self.I
        
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
        
    def plot(self, x, color = 'k'):
        pxs = np.array([x[0] + 0.5*self.l*np.cos(x[2]), x[0] - 0.5*self.l*np.cos(x[2])])
        pys = np.array([x[1] + 0.5*self.l*np.sin(x[2]), x[1] - 0.5*self.l*np.sin(x[2])])
        line = plt.plot(pxs, pys, marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
        return line

    def vis_traj(self, xs, dt = 0.1, axes_lim = [-5,5,-5,5]):
        T = len(xs)
        for x in xs:
            clear_output(wait=True)
            self.plot(x)
            plt.axes().set_aspect('equal')
            plt.axis(axes_lim)
            plt.show()
            time.sleep(dt)
            

class Quadcopter():
    def __init__(self, dt = 0.01, I = np.diag(np.array([2,2,4])), kd = 1, 
                 k = 1, L = 0.3, b = 1, m=1, g=9.81, Dx=12, Du=4):
        self.I = I #inertia
        self.kd = kd #friction
        self.k = k #motor constant
        self.L = L# distance between center and motor
        self.b = b # drag coefficient
        self.m = m # mass
        self.g = g
        self.Dx = Dx
        self.Du = Du
        self.dt = dt
        
    def thrust(self, inputs):
        T = np.array([0,0, self.k*np.sum(inputs)])
        return T

    def torques(self, inputs):
        tau = np.array([self.L*self.k*(inputs[0]-inputs[2]), self.L*self.k*(inputs[1]-inputs[3]), \
                        self.b*(inputs[0]-inputs[1] + inputs[2] - inputs[3])])
        return tau

    def acceleration(self, inputs, angles, xdot):
        gravity = np.array([0,0,-self.g])
        R = self.Rotation(angles)
        T = R.dot(self.thrust(inputs))
        Fd = -self.kd*xdot
        a = gravity + T/self.m + Fd
        return a

    def angular_acceleration(self, inputs, omega):
        tau = self.torques(inputs)
        omegadot = np.linalg.inv(self.I).dot(tau - np.cross(omega, self.I.dot(omega)))
        return omegadot

    def thetadot2omega(self, thetadot, theta):
        R = np.array([[1, 0, -np.sin(theta[1])], \
                     [0, np.cos(theta[0]), np.cos(theta[1])*np.sin(theta[0])], \
                     [0, -np.sin(theta[0]), np.cos(theta[1])*np.cos(theta[0])]])
        return R.dot(thetadot)

    def omega2thetadot(self, omega, theta):
        R = np.array([[1, 0, -np.sin(theta[1])], \
                     [0, np.cos(theta[0]), np.cos(theta[1])*np.sin(theta[0])], \
                     [0, -np.sin(theta[0]), np.cos(theta[1])*np.cos(theta[0])]])
        return np.linalg.inv(R).dot(omega)

    def Rotation(self, theta):
        c0,s0 = np.cos(theta[0]), np.sin(theta[0])
        c1,s1 = np.cos(theta[1]), np.sin(theta[1])
        c2,s2 = np.cos(theta[2]), np.sin(theta[2])

        R = np.array([[c0*c2 - c1*s0*s2, -c2*s0 - c0*c1*s2, s1 * s2], 
                     [c1*c2*s0 + c0*s2, c0*c1*c2-s0*s2, -c2*s1], 
                     [s0*s1, c0*s1, c1]])
        return R
        
    def set_init_state(self,x0):
        self.x0 = x0

    def compute_matrices(self,x,u, inc = 0.001):
        Dx, Du = len(x), len(u)
        A = np.zeros((Dx, Dx))
        B = np.zeros((Dx, Du))
        
        xnext = self.step(x, u)
        for i in range(Dx):
            xp, xm = x.copy(), x.copy()
            xp[i] += inc
            xnextp = self.step(xp, u)
            xm[i] -= inc
            xnextm = self.step(xm, u)
            diff = (xnextp - xnextm)/(2*inc)
            A[:,i] = diff
            
        for i in range(Du):
            up, um = u.copy(), u.copy()
            up[i] += inc
            xnextp = self.step(x, up)
            um[i] -= inc
            xnextm = self.step(x, um)
            diff = (xnextp - xnextm)/(2*inc)
            B[:,i] = diff
        
        return A,B
        
    def step(self, x, u, u_offset = None):
        if u_offset is None:
            u_mag = np.sqrt(9.81/4)
            u_offset = np.array([u_mag]*self.Du)**2 
        u_act = u_offset + u**2
        p, pdot, theta, thetadot = x[:3], x[3:6], x[6:9], x[9:]

        #step
        omega = self.thetadot2omega(thetadot, theta)
    
        a = self.acceleration(u_act, theta, pdot)
        omegadot = self.angular_acceleration(u_act, omega)
        omega = omega + self.dt*omegadot
        thetadot= self.omega2thetadot(omega, theta)
        theta = theta + self.dt*thetadot
        pdot = pdot + self.dt*a
        p = p + self.dt*pdot
        
        x_next = np.concatenate([p, pdot, theta, thetadot])
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
    def vis_quad(self, quadId, xs, dt = 0.05):
        for i,x in enumerate(xs):
            ori = euler2quat(xs[i,6:9], 'rzyz')
            p.resetBasePositionAndOrientation(quadId, xs[i,:3], ori)
            time.sleep(dt)
            p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=90, 
                                         cameraPitch=-20, cameraTargetPosition=xs[i,:3])
            
            
class QuadcopterCasadi():
    def __init__(self, dt = 0.01, I = np.diag(np.array([2,2,4])), kd = 1, 
                 k = 1, L = 0.3, b = 1, m=1, g=9.81, Dx=12, Du=4):
        self.I = I #inertia
        self.I_inv = np.linalg.inv(self.I)
        self.kd = kd #friction
        self.k = k #motor constant
        self.L = L# distance between center and motor
        self.b = b # drag coefficient
        self.m = m # mass
        self.g = g
        self.Dx = Dx
        self.Du = Du
        self.dt = dt
        self.gravity = np.array([0,0,-self.g])
        self.x = MX.sym('x', Dx)
        self.u = MX.sym('u', Du)
        
        #initialise
        self.def_step_func(self.x, self.u)
        self.def_jacobian()
        
    def thrust(self, inputs):
        T = vertcat(0,0, self.k*sum1(inputs))
        return T

    def torques(self, inputs):
        tau = vertcat(self.L*self.k*(inputs[0]-inputs[2]), self.L*self.k*(inputs[1]-inputs[3]), \
                        self.b*(inputs[0]-inputs[1] + inputs[2] - inputs[3]))
        return tau

    def acceleration(self, inputs, thetas, xdot):
        R = self.Rotation(thetas)
        T = mtimes(R,self.thrust(inputs))
        Fd = -self.kd*xdot
        a = self.gravity + T/self.m + Fd
        return a

    def angular_acceleration(self, inputs, omega):
        tau = self.torques(inputs)
        omegadot = mtimes(self.I_inv, tau - cross(omega, mtimes(self.I,omega)))
        return omegadot

    def thetadot2omega(self, thetadot, theta):
        R1 = vertcat(1,0,0)
        R2 = vertcat(0, cos(theta[0]), -sin(theta[0]))
        R3 = vertcat(-sin(theta[1]), cos(theta[1])*sin(theta[0]), cos(theta[1])*cos(theta[0]))
        R = horzcat(R1, R2, R3)
        return mtimes(R,thetadot)

    def omega2thetadot(self, omega, theta):
        R1 = vertcat(1,0,0)
        R2 = vertcat(0, cos(theta[0]), -sin(theta[0]))
        R3 = vertcat(-sin(theta[1]), cos(theta[1])*sin(theta[0]), cos(theta[1])*cos(theta[0]))
        R = horzcat(R1, R2, R3)
        return mtimes(casadi.inv(R), omega)

    def Rotation(self, theta):
        c0,s0 = cos(theta[0]), sin(theta[0])
        c1,s1 = cos(theta[1]), sin(theta[1])
        c2,s2 = cos(theta[2]), sin(theta[2])
        
        R1 = vertcat(c0*c2 - c1*s0*s2, c1*c2*s0 + c0*s2, s0*s1)
        R2 = vertcat(-c2*s0 - c0*c1*s2, c0*c1*c2-s0*s2, c0*s1 )
        R3 = vertcat( s1 * s2, -c2*s1, c1)
        R = horzcat(R1,R2,R3)
        return R
        
    def set_init_state(self,x0):
        self.x0 = x0
        
    def def_step_func(self, x, u, u_offset = None):
        if u_offset is None:
            u_mag = np.sqrt(9.81/4)
            u_offset = np.array([u_mag]*self.Du)**2 
        u_act = u_offset + u
        p, pdot, theta, thetadot = x[:3], x[3:6], x[6:9], x[9:]

        #step
        omega = self.thetadot2omega(thetadot, theta)
    
        a = self.acceleration(u_act, theta, pdot)
        omegadot = self.angular_acceleration(u_act, omega)
        omega = omega + self.dt*omegadot
        thetadot= self.omega2thetadot(omega, theta)
        theta = theta + self.dt*thetadot
        pdot = pdot + self.dt*a
        p = p + self.dt*pdot
        
        self.x_next = vertcat(p, pdot, theta, thetadot)
        self.step_fun = Function('step', [x, u], [self.x_next])

    def step(self, x, u):
        return np.array(self.step_fun(x,u)).flatten()
        
    def def_jacobian(self):
        self.A = jacobian(self.x_next, self.x)
        self.B = jacobian(self.x_next, self.u)
        
        self.A_val = Function('A', [self.x,self.u], [self.A])
        self.B_val = Function('B', [self.x,self.u], [self.B])

    def compute_matrices(self,x,u):
        A = np.array(self.A_val(x,u))
        B = np.array(self.B_val(x,u))
        return A, B
        
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
    def vis_traj(self, quadId, xs, dt = 0.05, camDist = 2.5, camYaw = 90, camPitch = -20, changeCamera = True):
        for i,x in enumerate(xs):
            ori = euler2quat(xs[i,6:9], 'rzyz')
            p.resetBasePositionAndOrientation(quadId, xs[i,:3], ori)
            time.sleep(dt)
            if changeCamera: p.resetDebugVisualizerCamera(cameraDistance=camDist, cameraYaw=camYaw, 
                                         cameraPitch= camPitch, cameraTargetPosition=xs[i,:3])
            
class ActionModelRobot(crocoddyl.ActionModelAbstract):
    def __init__(self, state, nu):
        crocoddyl.ActionModelAbstract.__init__(self, state, nu)
        
    def init_robot_sys(self,robot_sys, nr = 1):
        self.robot_sys = robot_sys
        self.Du = robot_sys.Du
        self.Dx = robot_sys.Dx
        self.Dr = nr
        
    def set_cost(self, cost_model):
        self.cost_model = cost_model
        
    def calc(self, data, x, u):
        #calculate the cost
        data.cost = self.cost_model.calc(x,u)
        
        #calculate the next state
        data.xnext = self.robot_sys.step(x,u)
        
    def calcDiff(self, data, x, u, recalc = False):
        if recalc:
            self.calc(data, x, u)

        #compute cost derivatives
        self.cost_model.calcDiff(x, u)
        data.Lx = self.cost_model.Lx.copy()
        data.Lxx = self.cost_model.Lxx.copy()
        data.Lu = self.cost_model.Lu.copy()
        data.Luu = self.cost_model.Luu.copy()
        
        #compute dynamic derivatives 
        A, B = self.robot_sys.compute_matrices(x,u)
        data.Fx = A.copy()
        data.Fu = B.copy()
        
    def createData(self):
        data = ActionDataRobot(self)
        return data

class ActionDataRobot(crocoddyl.ActionDataAbstract):
    def __init__(self, model):
        crocoddyl.ActionDataAbstract.__init__(self,model)
        
class TwoLinkRobot():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 4
        self.Du = 2
        self.l1 = 1.5
        self.l2 = 1
        
    def set_init_state(self,x0):
        self.x0 = x0
        
    def set_pref(self, p_ref):
        self.p_ref = p_ref
    
    def compute_matrices(self,x,u):
        A = np.eye(self.Dx)
        B = np.zeros((self.Dx,self.Du))
        
        A[0,2] = self.dt
        A[1,3] = self.dt
        
        B[2:,:] = np.eye(self.Du)
        
        self.A, self.B = A,B
        return A,B
    
    def compute_Jacobian(self, x):
        J = np.zeros((2, self.Dx))
        s1 = np.sin(x[0])
        c1 = np.cos(x[0])
        s12 = np.sin(x[0] + x[1])
        c12 = np.cos(x[0] + x[1])
        
        J[0,0] = -self.l1*s1 - self.l2*s12
        J[0,1] = - self.l2*s12
        J[1,0] =  self.l1*c1 + self.l2*c12
        J[1,1] =  self.l2*c12
        
        self.J = J
        return self.J
        
    def step(self, x, u):
        x_next = self.A.dot(x) + self.B.dot(u)
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
    def compute_ee(self,x):
        self.p1 = np.array([self.l1*np.cos(x[0]), self.l1*np.sin(x[0])])
        self.p2 = np.array([self.p1[0] + self.l2*np.cos(x[0] + x[1]), self.p1[1] + self.l2*np.sin(x[0] + x[1])])
        return self.p1, self.p2
    
    def plot(self, x, color='k'):
        self.compute_ee(x)
        
        line1 = plt.plot(np.array([0, self.p1[0]]),np.array([0, self.p1[1]]) , marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
        line2 = plt.plot(np.array([self.p1[0], self.p2[0]]),np.array([self.p1[1], self.p2[1]]) , marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
        xlim = [-1.5*(self.l1+self.l2), 1.5*(self.l1+self.l2)]
        plt.axes().set_aspect('equal')
        plt.axis(xlim+xlim)
        return line1,line2

    def plot_traj(self, xs, dt = 0.1):
        for x in xs:
            clear_output(wait=True)
            self.plot(x)
            plt.plot(self.p_ref[0], self.p_ref[1], '*')
            plt.show()
            time.sleep(self.dt)
            
class Manipulator():
    def __init__(self, rmodel, ee_id, dt = 0.01):
        self.dt = dt
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.Dx = rmodel.nq + rmodel.nv
        self.Du = rmodel.nv
        self.ee_id = ee_id
        
    def set_init_state(self,x0):
        self.x0 = x0
           
    def compute_matrices(self,x,u):
        A = np.eye(self.Dx)
        B = np.zeros((self.Dx,self.Du))
        
        A[:self.Dx/2,self.Dx/2:] = np.eye(self.Dx/2)*self.dt
        
        B[self.Dx/2:,:] = np.eye(self.Du)
        
        self.A, self.B = A,B
        return A,B
    
    def compute_Jacobian(self, x):
        q = x[:self.Dx/2]
        pin.forwardKinematics(self.rmodel,self.rdata,q)
        pin.updateFramePlacements(self.rmodel,self.rdata)
        pin.computeJointJacobians(self.rmodel, self.rdata, q)
        J = pin.getFrameJacobian(self.rmodel, self.rdata,self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.J = J[:3,:]
        return self.J
        
    def step(self, x, u):
        x_next = self.A.dot(x) + self.B.dot(u)
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
    def compute_ee(self,x):
        q = x[:self.Dx/2]
        pin.forwardKinematics(self.rmodel, self.rdata, q)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        pos, ori = self.rdata.oMf[self.ee_id].translation, self.rdata.oMf[self.ee_id].rotation
        return pos, ori

#     def plot_traj(self, xs, dt = 0.1):
#         for x in xs:
#             q = x[:self.Dx/2]
            
#             time.sleep(self.dt)