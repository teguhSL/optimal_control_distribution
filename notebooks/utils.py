from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv
import pybullet as p
import numpy as np
import pinocchio as pin
import transforms3d
import time
import pyscreenshot as ImageGrab

def set_ref(ddp, xs):
    #given the reference traj. xs, set the ddp to follow this xs
    T = len(xs)
    for i in range(T-1):
        ddp.problem.runningModels[i].cost_model.costs[0].set_ref(xs[i])
    ddp.problem.terminalModel.cost_model.costs[0].set_ref(xs[-1])
    
def set_Qref(ddp, Qref):
    #given the precision matrices Qref, set the ddp to use Qref in the cost function
    T = len(Qref)
    for i in range(T-1):
        ddp.problem.runningModels[i].cost_model.costs[0].Q = Qref[i]
    ddp.problem.terminalModel.cost_model.costs[0].Q = Qref[T-1]
    
def extract_ref(mu, sigma, Dx, T_hor, x, reg = 1e-6):
    #given the distribution N(mu,sigma), obtain the reference traj. distribution
    # as the marginal distribution at time t
    mu_ = mu.reshape(-1, Dx)
    T = mu.shape[0]
    
    #obtain the reference trajectory
    ref_x = subsample(np.vstack([x[None,:], mu_[:T_hor]]), T_hor+1)
    
    #obtain the reference precision
    Qs = np.zeros((T_hor+1, Dx, Dx))
    if T_hor < T:
        #if the horizon is within the remaining time steps, extract the marginal distribution
        for i in range(T_hor):
            Qs[i+1] = inv(sigma[Dx*i:Dx*(i+1), Dx*i:Dx*(i+1)]+ reg*np.eye(Dx))
    else:
        #if the horizon exceeds the remaining time steps
        for i in range(T):
            Qs[i+1] = inv(sigma[Dx*i:Dx*(i+1), Dx*i:Dx*(i+1)]+ reg*np.eye(Dx))
        for i in range(T, T_hor):
            Qs[i+1] = Qs[T]

    #the first Q does not affect the OCP and can be set as anything
    Qs[0] = Qs[1].copy()

    return ref_x, Qs

def calc_detail_cost(xs, us, ddp):
    rmodels = ddp.problem.runningModels
    cost_control = 0.
    cost_goal = 0.
    cost_state = 0.
    cost_col = []
    for i in range(len(xs)-1):
        costs = rmodels[i].cost_model.costs
        cost_state += costs[0].calc(xs[i], us[i])  
        cost_control +=  costs[1].calc(xs[i], us[i])
        cost_col += [costs[2].calc(xs[i], us[i])]
    cost_goal = ddp.problem.terminalModel.cost_model.calc(xs[-1], us[-1])
    return  cost_state, cost_control, cost_col, cost_goal

def lin_interpolate(state1, state2, n=1.):
    state_list = []
    for i in range(n+1):
        state_list.append(state1 + 1.*i*(state2-state1)/n)
    return state_list

def subsample(X,N):
    '''Subsample in N iterations the trajectory X. The output is a 
    trajectory similar to X with N points. '''
    nx  = X.shape[0]
    idx = np.arange(float(N))/(N-1)*(nx-1)
    hx  = []
    for i in idx:
        i0 = int(np.floor(i))
        i1 = int(np.ceil(i))
        di = i%1
        x  = X[i0,:]*(1-di) + X[i1,:]*di
        hx.append(x)
    return np.vstack(hx)


def rotvec2mat(x):
    return R.as_matrix(R.from_rotvec(x))


def save_screenshot(x,y,w,h,file_name, to_show='False'):
    # part of the screen
    im=ImageGrab.grab(bbox=(x,y,w,h))
    if to_show:
        im.show()
    # save to file
    im.save(file_name)
    return im

def get_pb_config(q):
    """
    Convert tf's format of 'joint angles + base position' to
    'base_pos + base_ori + joint_angles' according to pybullet order
    """
    joint_angles = q[:28]
    #qnew = np.concatenate([q[28:31], euler2quat(q[-3:]),
    #qnew = np.concatenate([np.array([0,0,q[28]]), euler2quat(q[-3:]),
    qnew = np.concatenate([q[28:31], euler2quat(np.array([0,0,0])),
                           joint_angles[-6:], joint_angles[-12:-6], 
                          joint_angles[:2], joint_angles[9:16], 
                          joint_angles[2:9]])
    return qnew

def get_tf_config(q):
    """
    Convert 'base_pos + base_ori + joint_angles' according to pybullet order
    to tf's format of 'joint angles + base position'
    
    """
    joint_angles = q[7:]
    qnew = np.concatenate([joint_angles[12:14], joint_angles[-7:], 
                          joint_angles[-14:-7], joint_angles[6:12], 
                          joint_angles[:6], q[:3]])
    return qnew


def normalize(x):
    return x/np.linalg.norm(x)
        
def set_q(q, robot_id, joint_indices, set_base = False):
    if set_base:
        localInertiaPos = np.array(p.getDynamicsInfo(robot_id,-1)[3])
        q_root = q[0:7]
        ori = q_root[3:]
        Rbase = np.array(p.getMatrixFromQuaternion(ori)).reshape(3,3)
        shift_base = Rbase.dot(localInertiaPos)
        pos = q_root[:3]+shift_base
        p.resetBasePositionAndOrientation(robot_id,pos,ori)
        q_joint = q[7:]
    else:
        q_joint = q
    
    #set joint angles
    for i in range(len(q_joint)):
        p.resetJointState(robot_id, joint_indices[i], q_joint[i])


def vis_traj(qs, vis_func, dt=0.1):
    for q in qs:
        vis_func(q)
        time.sleep(dt)


def get_joint_limits(robot_id, indices):
    lower_limits = []
    upper_limits = []
    for i in indices:
        info = p.getJointInfo(robot_id, i)
        lower_limits += [info[8]]
        upper_limits += [info[9]]
    limits = np.vstack([lower_limits, upper_limits])
    return limits

def computeJacobian(rmodel,rdata,ee_frame_id,q):
    pin.forwardKinematics(rmodel,rdata,q)
    pin.updateFramePlacements(rmodel,rdata)
    pin.computeJointJacobians(rmodel, rdata, q)
    J = pin.getFrameJacobian(rmodel, rdata,ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return J[:,:7]

def computePose(rmodel, rdata, ee_frame_id, q):
    pin.forwardKinematics(rmodel, rdata, q)
    pin.updateFramePlacements(rmodel, rdata)
    pos, ori = rdata.oMf[ee_frame_id].translation, rdata.oMf[ee_frame_id].rotation
    return pos,ori
    
def check_joint_limits(q, joint_limits):
    """
    Return True if within the limit
    """
    upper_check = False in ((q-joint_limits[0]) > 0)
    lower_check = False in ((joint_limits[1]-q) > 0)
    if upper_check or lower_check:
        return False
    else:
        return True
    
def calc_dist_limit(q, joint_limits):
    lower_error = joint_limits[0]-q
    lower_check = (lower_error > 0)
    lower_error = lower_error*lower_check
    upper_error = q-joint_limits[1]
    upper_check = (upper_error > 0)
    upper_error = upper_error*upper_check
    error = lower_error-upper_error
    return error
    
def mat2euler(rot, axes = 'rzyx'):
    return np.array(transforms3d.euler.mat2euler(rot, axes = axes))

def euler2quat(rpy, axes='sxyz'):
    #euler sxyz: used by Manu's codes
    return rectify_quat(transforms3d.euler.euler2quat(*rpy, axes=axes))

def rectify_quat(quat):
    #transform from transforms3d format (w,xyz) to pybullet and pinocchio (xyz, w)
    quat_new = np.concatenate([quat[1:], quat[0:1]])
    return quat_new

def mat2w(rot):
    rot_aa = pin.AngleAxis(rot)
    return rot_aa.angle*rot_aa.axis

def w2quat(q):
    angle = np.linalg.norm(q)
    if abs(angle) < 1e-7:
        ax = np.array([1,0,0])
    else:
        ax, angle = normalize(q), np.linalg.norm(q)
    w = p.getQuaternionFromAxisAngle(ax, angle)
    return np.array(w)

def quat2w(q):
    ax, angle = p.getAxisAngleFromQuaternion(q)
    return np.array(ax)*angle

def w2mat(w):
    angle = np.linalg.norm(w)
    if abs(angle) < 1e-7:
        ax = np.array([1,0,0])
    else:
        ax, angle = w/angle, angle
    R = pin.AngleAxis.toRotationMatrix(pin.AngleAxis(angle, ax))
    return R

def get_link_base(robot_id, frame_id):
    '''
    Obtain the coordinate of the link frame, according to the convention of pinocchio (at the link origin, 
    instead of at the COM as in pybullet)
    '''
    p1 = np.array(p.getLinkState(robot_id,frame_id)[0])
    ori1 = np.array(p.getLinkState(robot_id,frame_id)[1])
    R1 = np.array(p.getMatrixFromQuaternion(ori1)).reshape(3,3)
    p2 = np.array(p.getLinkState(robot_id,frame_id)[2])
    return  p1 - R1.dot(p2), ori1

    
def create_primitives(shapeType=2, rgbaColor=[1, 1, 0, 1], pos = [0, 0, 0], radius = 1, length = 2, halfExtents = [0.5, 0.5, 0.5], baseMass=1, basePosition = [0,0,0]):
    visualShapeId = p.createVisualShape(shapeType=shapeType, rgbaColor=rgbaColor, visualFramePosition=pos, radius=radius, length=length, halfExtents = halfExtents)
    collisionShapeId = p.createCollisionShape(shapeType=shapeType, collisionFramePosition=pos, radius=radius, height=length, halfExtents = halfExtents)
    bodyId = p.createMultiBody(baseMass=baseMass,
                      baseInertialFramePosition=[0, 0, 0],
                      baseVisualShapeIndex=visualShapeId,
                      baseCollisionShapeIndex=collisionShapeId,    
                      basePosition=basePosition,
                      useMaximalCoordinates=True)
    return visualShapeId, collisionShapeId, bodyId

    
#### Code to modify concave objects in pybullet
#name_in =  rl.datapath + '/urdf/bookcase_old.obj'
#name_out = rl.datapath + '/urdf/bookcase.obj'
#name_log = "log.txt"
#p.vhacd(name_in, name_out, name_log, alpha=0.04,resolution=10000000 )

def Rot_z(angle):
    w = np.array([0,0,angle])
    Rz = w2mat(w)
    return Rz
