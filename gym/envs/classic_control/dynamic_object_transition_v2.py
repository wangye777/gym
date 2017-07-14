# agent applies force in 8 different directions

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

def _isIn(pos, region):
    if pos[0] >= region[0] and pos[0] <= region[1] and pos[1] >= region[2] and pos[1] <= region[3]:
        return True
    else:
        return False

def _isInCircle(pos, circle):
    if _distance(pos, [circle[0],circle[1]]) <= circle[2]:
        return True
    else:
        return False

def _distance(pos1, pos2):
    return math.sqrt(math.pow(pos1[0]-pos2[0], 2) + math.pow(pos1[1]-pos2[1], 2))

def _norm(vec):
    return np.linalg.norm(vec,2)

class DynamicObjectTransitionV2Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.dT = 0.1 # time step

        #self.min_f = -1.0
        #self.max_f = 1.0
        self.min_fmag = 0
        self.max_fmag = 1.0
        self.min_fdir = 0.0
        self.max_fdir = 2*math.pi - 0.01
        self.max_vel = 10.0 # max velocity of the object

        # # size
        # self.region = [0, 100, 0, 100] # l,r,b,u
        # self.pos_reg = [10, 90, 10, 90]
        # self.goal_reg = [10, 90, 10, 90]
        # self.num_obsts = 4
        # self.goal_rad = 10
        # self.obst_reg = [10, 90, 10, 90]
        # self.obst_rad = 6


        # size
        self.region = [0, 100/2, 0, 100/2] # l,r,b,u
        self.pos_reg = [10/2, 90/2, 10/2, 90/2]
        self.goal_reg = [10/2, 90/2, 10/2, 90/2]
        self.goal_rad = 10/2
        self.num_obsts = 6 #6
        self.obst_reg = [10/2, 90/2, 10/2, 90/2]
        self.obst_rad = 6/2
        

        self.friction = 0.8
        self.mass = 1.0 # mass of the object

        # (pos_x, vel_x, pos_y, vel_y, goal_x, goal_y, goal_rad, obst_x1, obst_y1, obst_rad1)
        self.low_state = np.array([self.region[0], self.region[2], -self.max_vel, -self.max_vel, self.goal_reg[0], self.goal_reg[2], self.goal_rad])
        for i in xrange(self.num_obsts):
            self.low_state = np.append(self.low_state, np.array([self.obst_reg[0], self.obst_reg[2], self.obst_rad]))
        self.high_state = np.array([self.region[1], self.region[3], self.max_vel, self.max_vel, self.goal_reg[1], self.goal_reg[3], self.goal_rad])
        for i in xrange(self.num_obsts):
            self.high_state = np.append(self.high_state, np.array([self.obst_reg[1], self.obst_reg[3], self.obst_rad]))

        self.min_action = np.tile([self.min_fmag, self.min_fdir], 4)
        self.max_action = np.tile([self.max_fmag, self.max_fdir], 4)       

        self.agent_num = 4

        self.viewer = None

        self.observation_space = spaces.Box(self.low_state, self.high_state)
        self.action_space = spaces.Box(self.min_action, self.max_action)

        self.action = np.zeros([8])

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        #self.state = np.array([self.np_random.uniform(low=0, high=30), self.np_random.uniform(low=0, high=40)])
        #self.state = np.array([self.np_random.uniform(low=50, high=70), self.np_random.uniform(low=0, high=40)])
        self.state = np.array([self.np_random.uniform(low=self.pos_reg[0], high=self.pos_reg[1]), \
            self.np_random.uniform(low=self.pos_reg[2], high=self.pos_reg[3]), 0.0, 0.0])

        self.goal = np.array([self.np_random.uniform(low=self.goal_reg[0]+self.goal_rad, high=self.goal_reg[1]-self.goal_rad), \
            self.np_random.uniform(low=self.goal_reg[2]+self.goal_rad, high=self.goal_reg[3]-self.goal_rad), self.goal_rad])
        #if _distance([self.state[0],self.state[2]], [self.goal[0],self.goal[1]]) < 30: self._reset() #no init around goal

        self.state = np.append(self.state,self.goal)
        self.obstacles = []
        for i in xrange(self.num_obsts):
            obst = np.array([self.np_random.uniform(low=self.obst_reg[0], high=self.obst_reg[1]), \
                self.np_random.uniform(low=self.obst_reg[2]+self.obst_rad, high=self.obst_reg[3]-self.obst_rad), self.obst_rad])
            self.obstacles.append(obst)
            self.state = np.append(self.state, obst)
        
        return np.array(self.state)

    def _set_state(self, state):
        self.state = np.array(state)

    def _get_state(self):
        return self.state

    def _configure(self, info):
        self._set_state(info['state'])

    def _step(self, raw_action):
        # raw_action = [f1_mag, f1_theta, f2_mag, f2_theta, ...]
        #print self.state
        action = np.clip(raw_action, self.min_action, self.max_action)
        self.action = action
        position = [self.state[0], self.state[1]]
        #print("position=",position)
        velocity = [self.state[2], self.state[3]]

        ##### calculate the sum force from the robots
        f_x = 0.0 # sum force of the group in x direction
        f_y = 0.0
        for i in range(self.agent_num):
            f_x += action[2*i] * math.cos(action[2*i+1])
            f_y += action[2*i] * math.sin(action[2*i+1])


        f_sig = math.sqrt(math.pow(f_x, 2) + math.pow(f_y, 2)) # magnitude of the robot force

        ##### deal with planar object dynamics
        if ( (f_sig <= self.friction) and (_norm(velocity) <= 0.1) ):
            # object is stationary and the input force is smaller than the friction
            velocity[0] = 0.0
            velocity[1] = 0.0
            # position doesn't change
        elif( (f_sig > self.friction) and (_norm(velocity) <= 0.1) ):
            # object is stationary, but input force is larger than friction
            # in this case, friction has the opposite direction to the input force
            friction_x = - self.friction * f_x / _norm(np.array([f_x, f_y])) # friction force has negative direction w.r.t. object velocity
            friction_y = - self.friction * f_y / _norm(np.array([f_x, f_y]))
            fsum_x = f_x + friction_x # sum force by combining robot forces and friction
            fsum_y = f_y + friction_y   
            #print("not moving fsum_x={} fsum_y={}".format(fsum_x,fsum_y))
            velocity[0] = velocity[0] + fsum_x / self.mass * self.dT
            velocity[1] = velocity[1] + fsum_y / self.mass * self.dT
        else:
            # in all other cases, friction has the opposite direction to the velocity
            friction_x = - self.friction * velocity[0] / _norm(velocity) # friction force has negative direction w.r.t. object velocity
            friction_y = - self.friction * velocity[1] / _norm(velocity)
            fsum_x = f_x + friction_x # sum force by combining robot forces and friction
            fsum_y = f_y + friction_y
            #print("moving fsum_x={} fsum_y={}".format(fsum_x,fsum_y))
            velocity[0] = velocity[0] + fsum_x / self.mass * self.dT
            velocity[1] = velocity[1] + fsum_y / self.mass * self.dT
        position[0] = position[0] + velocity[0] * self.dT
        position[1] = position[1] + velocity[1] * self.dT

        ##### check goal and assign rewards
        is_in_goal = _isInCircle(position, self.goal)
        is_in_obstacles = False
        for obstacle in self.obstacles:
            if _isInCircle(position, obstacle) is True:
                is_in_obstacles = True
                break
        if is_in_goal: is_in_obstacles = False

        if (position[0] < self.region[0]): position[0] = self.region[0]
        if (position[0] > self.region[1]): position[0] = self.region[1]
        if (position[1] < self.region[2]): position[1] = self.region[2]
        if (position[1] > self.region[3]): position[1] = self.region[3]        

        done = (is_in_goal and _norm(velocity) < 0.5) #or is_in_obstacles
    
        goal_x = self.goal[0]
        goal_y = self.goal[1]
        dist1 = _distance([self.state[0],self.state[1]], [goal_x, goal_y])
        dist2 = _distance(position, [goal_x, goal_y])

        #reward = 0
        reward = -1
        if is_in_goal:
            reward += 2.0
        if is_in_obstacles:
            reward = -10.0
        if done and (not is_in_obstacles):
            reward = 30
        
        if dist1 - dist2 <= 0.05: reward -= 1 #dese reward

        #if reward == 0.0: reward = -1
        #print("\naction={}, f_sig={}, reward={}, dist1={}, dist2={}".format(action, f_sig, reward, dist1, dist2))

        self.state[0:4] = np.array([position[0], position[1], velocity[0], velocity[1]])

        return self.state, reward, done, {"done":done}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        scale = 10
        
        screen_width = (self.region[1] - self.region[0]) * scale
        screen_height = (self.region[3] - self.region[2]) * scale

        world_width = self.region[1] - self.region[0]
        scale = screen_width/world_width
        objwidth=40
        objheight=40

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            #render obstacles
            self.render_obsts = []
            self.obsts_trans_list = []
            for obstacle in self.obstacles:
                obst_trans = rendering.Transform()
                obst = rendering.make_circle(obstacle[2]*scale)
                obst.add_attr(obst_trans)
                obst.set_color(0,0,0)
                self.viewer.add_geom(obst)
                self.obsts_trans_list.append(obst_trans)
                self.render_obsts.append(obst)
            
            #render goal
            self.goal_trans = rendering.Transform()
            self.goal_ren = rendering.make_circle(self.goal[2]*scale)
            self.goal_ren.add_attr(self.goal_trans)
            self.goal_ren.set_color(0,255,0)
            self.viewer.add_geom(self.goal_ren)
            #print("goal[0]={} goal[1]={}".format(self.goal[0],self.goal[1]))

            #render arrows
            fname = path.join(path.dirname(__file__), "assets/arrow.png")
            self.img_list = []
            self.imgtrans_list = []
            for id in xrange(self.agent_num):
                img = rendering.Image(fname, 20., 40.)
                #print("img={}".format(img))
                imgtrans = rendering.Transform()
                img.add_attr(imgtrans)
                self.img_list.append(img)
                self.imgtrans_list.append(imgtrans)
            #print("self.img_list={}, self.imgtrans_list={}".format(self.img_list,self.imgtrans_list))

            self.objtrans = rendering.Transform()
            l,r,t,b = -objwidth/2, objwidth/2, -objheight/2, objheight/2
            self.obj = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])            
            self.obj.add_attr(self.objtrans)
            self.obj.set_color(255,0,0)
            self.viewer.add_geom(self.obj)

        self.scales = []
        self.rotations = []
        for id in xrange(self.agent_num):
            self.scales.append(np.abs(self.action[id*2]))
            self.rotations.append(self.action[id*2+1]-math.pi/2) #anti-clockwise
        self.arrow_offsets = [[30,0],[0,-30],[-30,0],[0,30]] * scale
        
        for id in xrange(self.agent_num):
            self.viewer.add_onetime(self.img_list[id])
            self.imgtrans_list[id].set_translation(self.state[0]*scale+self.arrow_offsets[id][0],\
                self.state[1]*scale+self.arrow_offsets[id][1]) #follow object #arrow vis V1
            #self.imgtrans_list[id].set_translation(self.state[0]*scale,self.state[2]*scale) #arrow vis V2, follow object
            #self.imgtrans_list[id].set_translation(100+60*id, 200) #arrow vis V3, fixed position
            self.imgtrans_list[id].set_rotation(self.rotations[id]) # rotation
            self.imgtrans_list[id].scale = (self.scales[id],self.scales[id])
        
        for i, obst_trans in enumerate(self.obsts_trans_list):
            obst_trans.set_translation(self.obstacles[i][0]*scale,self.obstacles[i][1]*scale)
        self.goal_trans.set_translation(self.goal[0]*scale,self.goal[1]*scale)
        self.objtrans.set_translation(self.state[0]*scale, self.state[1]*scale)
        #self.objtrans.set_translation(100, 200)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
