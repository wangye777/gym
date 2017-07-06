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

def _distance(pos1, pos2):
    return math.sqrt(math.pow(pos1[0]-pos2[0], 2) + math.pow(pos1[1]-pos2[1], 2))

class ObjectTransitionEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_f = -1.0
        self.max_f = 1.0
        self.region = [0, 80, 0, 40]
        # self.min_x = 0
        # self.max_x = 80
        # self.min_y = 0
        # self.max_y = 40

        self.obstacles = []
        self.obstacles.append([40, 50, 15, 25])
        
        self.goal = [64, 70, 17, 23]

        self.friction = 0.8

        self.low_state = np.array([self.region[0], self.region[2]])
        self.high_state = np.array([self.region[1], self.region[3]])

        self.min_action = np.repeat(self.min_f, 4)
        self.max_action = np.repeat(self.max_f, 4)       

        self.agent_num = 4

        self.viewer = None

        self.observation_space = spaces.Box(self.low_state, self.high_state)
        self.action_space = spaces.Box(self.min_action, self.max_action)

        self.action = np.zeros([4])

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _set_state(self, state):
        self.state = np.array(state)

    def _step(self, raw_action):
        action = np.clip(raw_action, self.min_f, self.max_f)
        self.action = action
        position = [self.state[0], self.state[1]]
        f_x = action[0] + action[2]
        f_y = action[1] + action[3]
        f_sig = math.sqrt(math.pow(f_x, 2) + math.pow(f_x, 2))
    
        if f_sig > self.friction:
            f_x = f_x * (f_sig - self.friction) / f_sig
            f_y = f_y * (f_sig - self.friction) / f_sig
        else:
            f_x = 0.0
            f_y = 0.0
    
    
        v_x = f_x * 4
        v_y = f_y * 4
        position[0] += v_x
        position[1] += v_y

        is_in_goal = _isIn(position, self.goal)
        is_in_obstacles = False
        for obstacle in self.obstacles:
            if _isIn(position, obstacle) is True:
                is_in_obstacles = True
                break

        if (position[0] < self.region[0]): position[0] = self.region[0]
        if (position[0] > self.region[1]): position[0] = self.region[1]
        if (position[1] < self.region[2]): position[1] = self.region[2]
        if (position[1] > self.region[3]): position[1] = self.region[3]        

        done = is_in_goal or is_in_obstacles
    
        goal_x = (self.goal[0]+self.goal[1])/2
        goal_y = (self.goal[2]+self.goal[3])/2
        dist1 = _distance([self.state[0],self.state[1]], [goal_x, goal_y])
        dist2 = _distance(position, [goal_x, goal_y])

        reward = -1
        if is_in_goal:
            reward += 100.0
        if is_in_obstacles:
            reward -= 100.0
        if dist1 - dist2 < 0: reward -= 1
        #reward += np.sign(dist1 - dist2)
        #if reward == 0.0: reward = -1
        #print("\naction={}, f_sig={}, reward={}, dist1={}, dist2={}".format(action, f_sig, reward, dist1, dist2))

        self.state = np.array(position)
        return self.state, reward, done, {"done":done}

    def _reset(self):
        self.state = np.array([self.np_random.uniform(low=0, high=30), self.np_random.uniform(low=0, high=40)])
        #self.state = np.array([self.np_random.uniform(low=50, high=70), self.np_random.uniform(low=0, high=40)])
        return np.array(self.state)

    def _set_state(self, init_state):
        self.state = np.array(init_state)

    def _get_state(self):
        return self.state

    def _configure(self, info):
        self._set_state(info['state'])

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 800
        screen_height = 400

        world_width = self.region[1] - self.region[0]
        scale = screen_width/world_width
        objwidth=40
        objheight=40

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.objtrans = rendering.Transform()

            l,r,t,b = -objwidth/2, objwidth/2, -objheight/2, objheight/2
            self.obj = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])            
            self.obj.add_attr(self.objtrans)
            self.obj.set_color(255,0,0)
            self.viewer.add_geom(self.obj)

            self.render_obsts = []
            for obstacle in self.obstacles:
                l,r,t,b = obstacle[0]*scale, obstacle[1]*scale, obstacle[2]*scale, obstacle[3]*scale
                render_obst = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                render_obst.set_color(0,0,0)
                self.viewer.add_geom(render_obst)
                self.render_obsts.append(render_obst)

            l,r,t,b = self.goal[0]*scale, self.goal[1]*scale, self.goal[2]*scale, self.goal[3]*scale
            self.goal_reg = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.goal_reg.set_color(0,255,0)
            self.viewer.add_geom(self.goal_reg)

            fname = path.join(path.dirname(__file__), "assets/arrow.png")
            self.img_list = []
            self.imgtrans_list = []
            for id in xrange(self.agent_num):
                img = rendering.Image(fname, 40., 80.)
                print("img={}".format(img))
                imgtrans = rendering.Transform()
                img.add_attr(imgtrans)
                self.img_list.append(img)
                self.imgtrans_list.append(imgtrans)
            print("self.img_list={}, self.imgtrans_list={}".format(self.img_list,self.imgtrans_list))


        self.objtrans.set_translation(self.state[0]*scale, self.state[1]*scale)

        self.scales = []
        self.rotations = []
        for id in xrange(self.agent_num):
            self.scales.append(np.abs(self.action[id]))
            if id % 2 == 0:
                if self.action[id] >= 0: self.rotations.append(-math.pi/2) #anti-clockwise
                else: self.rotations.append(math.pi/2)
            else:
                if self.action[id] >= 0: self.rotations.append(0)
                else: self.rotations.append(math.pi)
        
        for id in xrange(self.agent_num):
            self.viewer.add_onetime(self.img_list[id])
            self.imgtrans_list[id].set_translation(self.state[0]*scale, self.state[1]*scale) #follow object
            self.imgtrans_list[id].set_translation(100+80*id, 200) #different position for different agents
            self.imgtrans_list[id].set_rotation(self.rotations[id]) # rotation
            self.imgtrans_list[id].scale = (self.scales[id],self.scales[id])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
