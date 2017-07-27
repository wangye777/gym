# no dynamics

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

class DynamicObjectTransitionV3Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.dT = 0.1 # time step

        #self.min_f = -1.0
        #self.max_f = 1.0
        self.min_fdir = 0.0
        self.max_fdir = 2*math.pi - 0.01
        self.max_vel = 10.0 # max velocity of the object

        # # size
        # self.region = [0, 100, 0., 100] # l,r,b,u
        # self.pos_reg = [10, 90, 10, 90]
        # self.goal_reg = [10, 90, 10, 90]
        # self.num_obsts = 4
        # self.goal_rad = 10
        # self.obst_reg = [10, 90, 10, 90]
        # self.obst_rad = 6


        # size
        self.region = [0, 100/2, 0, 100/2] # l,r,b,u
        self.pos_reg = [0, 100/2, 0, 100/2]
        self.goal_reg = [10/2, 90/2, 10/2, 90/2]
        # self.goal_reg = [50/2-0.1, 50/2+0.1, 50/2-0.1, 50/2+0.1]
        self.goal_rad = 10./2
        self.num_obsts = 6 #6
        self.obst_reg = [0, 100/2, 0, 100/2]
        self.obst_rad = 6./2
        

        self.friction = 0.8
        self.mass = 1.0 # mass of the object

        # (pos_x, pos_y, vel_x, vel_y, goal_x, goal_y, goal_rad, obst_x1, obst_y1, obst_rad1)
        self.low_state = np.array([self.region[0], self.region[2], -self.max_vel, -self.max_vel, self.region[0], self.region[2], 0.])
        for i in xrange(self.num_obsts):
            self.low_state = np.append(self.low_state, np.array([self.region[0], self.region[2], 0.]))
        self.high_state = np.array([self.region[1], self.region[3], self.max_vel, self.max_vel, self.region[1], self.region[3], 20.])
        for i in xrange(self.num_obsts):
            self.high_state = np.append(self.high_state, np.array([self.region[1], self.region[3], 20.]))

        self.agent_num = 1

        self.min_action = np.tile([-self.max_vel, -self.max_vel], self.agent_num)
        self.max_action = np.tile([self.max_vel, self.max_vel], self.agent_num)  

        self.viewer = None

        self.observation_space = spaces.Box(self.low_state, self.high_state)
        self.action_space = spaces.Box(self.min_action, self.max_action)

        self.action = np.zeros([self.agent_num * self.action_space.shape[0]])

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

        self.goal = np.array([self.np_random.uniform(low=self.goal_reg[0], high=self.goal_reg[1]), \
                    self.np_random.uniform(low=self.goal_reg[2], high=self.goal_reg[3]), self.goal_rad])
        # self.goal = np.array([self.np_random.uniform(low=self.goal_reg[0]+self.goal_rad, high=self.goal_reg[1]-self.goal_rad), \
        #     self.np_random.uniform(low=self.goal_reg[2]+self.goal_rad, high=self.goal_reg[3]-self.goal_rad), self.goal_rad])
        #if _distance([self.state[0],self.state[2]], [self.goal[0],self.goal[1]]) < 30: self._reset() #no init around goal

        self.state = np.append(self.state,self.goal)
        self.obstacles = []
        
        for i in xrange(self.num_obsts):
            obst = np.array([self.np_random.uniform(low=self.obst_reg[0], high=self.obst_reg[1]), \
                self.np_random.uniform(low=self.obst_reg[2]+self.obst_rad, high=self.obst_reg[3]-self.obst_rad), self.obst_rad])
            self.obstacles.append(obst)
            self.state = np.append(self.state, obst)
        
        # obsts = np.array([18, 35, self.obst_rad, 18, 15, self.obst_rad, 32, 15, self.obst_rad,\
        #      32, 35, self.obst_rad, 10, 25, self.obst_rad, 40, 25, self.obst_rad])
        # for i in xrange(self.num_obsts): 
        #     self.obstacles.append(obsts[i*3:i*3+3])
        #     self.state = np.append(self.state, obsts[i*3:i*3+3])        
        
        return np.array(self.state)

    def _set_state(self, state):
        self.state = np.array(state)

    def _get_state(self):
        return self.state

    def _configure(self, info):
        self._set_state(info['state'])

    def _step(self, raw_action):
        # raw_action = [f1_mag, f1_theta, f2_mag, f2_theta, ...]
        action = np.clip(raw_action, self.min_action, self.max_action)
        self.action = action
        position = [self.state[0], self.state[1]]
        velocity = [self.action[0], self.action[1]]

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

        done = (is_in_goal and _norm(velocity) < 6) #or is_in_obstacles
    
        goal_x = self.goal[0]
        goal_y = self.goal[1]
        dist1 = _distance([self.state[0],self.state[1]], [goal_x, goal_y])
        dist2 = _distance(position, [goal_x, goal_y])

        #reward = 0
        reward = -1
        if dist1 - dist2 <= 0.05: reward -= 1 #dese reward
        if is_in_goal:
            reward = 1.0
        if is_in_obstacles:
            reward = -5.0
        if done and (not is_in_obstacles):
            reward = 5

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

            self.objtrans = rendering.Transform()
            l,r,t,b = -objwidth/2, objwidth/2, -objheight/2, objheight/2
            self.obj = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])            
            self.obj.add_attr(self.objtrans)
            self.obj.set_color(255,0,0)
            self.viewer.add_geom(self.obj)

        
        for i, obst_trans in enumerate(self.obsts_trans_list):
            obst_trans.set_translation(self.obstacles[i][0]*scale,self.obstacles[i][1]*scale)
        self.goal_trans.set_translation(self.goal[0]*scale,self.goal[1]*scale)
        self.objtrans.set_translation(self.state[0]*scale, self.state[1]*scale)
        #self.objtrans.set_translation(100, 200)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
