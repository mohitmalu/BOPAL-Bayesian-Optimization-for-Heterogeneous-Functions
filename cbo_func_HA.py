
import numpy as np
np.float_ = np.float64  ## This is a workaround for staliro library which uses np.float_ instead of np.float64
import matplotlib.pyplot as plt
import polytope as pc
from scipy.integrate import odeint
from shapely.geometry import Polygon, Point
from staliro.specifications.rtamt import parse_dense as RTAMTDense
from staliro import Trace

class HA:

    def __init__(self, totalTime = 2, tStop = 0.05, tInc = 0.01):
        set_1_A = np.array([
                [1, 0],
                [-1,0],
                [0, 1],
                [0,-1],
            ])

        set_1_b = np.array([1, 1, 1, 1])
        set_1_p = pc.Polytope(set_1_A, set_1_b)

        set_2_A = np.array([
                        [1, 0],
                        [-1,0],
                        [0, 1],
                        [0,-1],
                    ])

        set_2_b = np.array([0.95, -0.85, 0.95, -0.85])
        set_2_p = pc.Polytope(set_2_A, set_2_b)

        self.green_set = set_1_p.diff(set_2_p)
        self.yellow_set = set_2_p

        result_state_set_1_A = np.array([
                    [1, 0],
                    [-1,0],
                    [0, 1],
                    [0,-1],
                ])
        result_state_set_1_b = np.array([-1.4, 1.8, -1.4, 1.6])

        result_state_set_2_A = np.array([
                            [1, 0],
                            [-1,0],
                            [0, 1],
                            [0,-1],
                        ])
        result_state_set_2_b = np.array([4.1, -3.7, -1.4, 1.6])
        self.result_state_set_1_polytope = pc.Polytope(result_state_set_1_A, result_state_set_1_b)
        self.result_state_set_2_polytope = pc.Polytope(result_state_set_2_A, result_state_set_2_b)

        self.yellow_polygon_def = Polygon([(0.85,0.85), (0.85,0.95),(0.95,0.95),(0.95,0.85)])
        self.totalTime = totalTime
        self.tStop = tStop
        self.tInc = tInc

    def _updatemember(self, x: float, y: float):
        pass

    def evaluate_state(self, init_point):
        
        time, traj = self._generate_traj(init_point)   

        # phi_unsafe_x = "x_pos <= 0.95 and x_pos >= 0.85"
        # phi_unsafe_y = "y_pos <= 0.95 and y_pos >= 0.85"
        # phi_unsafe = f"G[0,2] (not (({phi_unsafe_x}) and ({phi_unsafe_y})))"
        # specification_unsafe = RTAMTDense(phi_unsafe,  {"x_pos": 0, "y_pos": 1})
        # dist_1 = specification_unsafe.evaluate(traj.T, time)

        dist_2 = self.yellow_polygon_def.distance(Point(init_point))
        
        return dist_2

    def next_target_state(self, init_point):
        
        time, traj = self._generate_traj(init_point)   

        phi_unsafe_x = "x_pos <= 0.95 and x_pos >= 0.85"
        phi_unsafe_y = "y_pos <= 0.95 and y_pos >= 0.85"
        phi_unsafe = f"G[0,2] (not (({phi_unsafe_x}) and ({phi_unsafe_y})))"
        specification_unsafe = RTAMTDense(phi_unsafe, {"x_pos": 0, "y_pos": 1})
        dist_1 = max(0.0, specification_unsafe.evaluate(traj.T, time))

        dist_2 = self.yellow_polygon_def.distance(Point(init_point))
        
        return (dist_1, dist_2)

    def get_hybrid_distance(self, init_point):
        
        time, traj = self._generate_traj(init_point)   

        phi_unsafe_x = "x_pos <= 0.95 and x_pos >= 0.85"
        phi_unsafe_y = "y_pos <= 0.95 and y_pos >= 0.85"
        phi_unsafe = f"G[0,2] (not (({phi_unsafe_x}) and ({phi_unsafe_y})))"
        specification_unsafe = RTAMTDense(phi_unsafe, {"x_pos": 0, "y_pos": 1})
        dist_1 = specification_unsafe.evaluate(traj.T, time)

        dist_2 = self.yellow_polygon_def.distance(Point(init_point))
        
        return (dist_1, dist_2)

    def get_robustness(self, init_point):
        
        time, traj = self._generate_traj(init_point)   
        traj = traj.T
        # print(traj.shape)
        # print(time.shape)
        phi_1_x = "x_pos <= -1.4 and x_pos >= -1.8"
        phi_1_y = "y_pos <= -1.4 and y_pos >= -1.6"
        phi_1 = f"({phi_1_x}) and ({phi_1_y})"


        phi_2_x = "x_pos <= 4.1 and x_pos >= 3.7"
        phi_2_y = "y_pos <= -1.4 and y_pos >= -1.6"
        phi_2 = f"({phi_2_x}) and ({phi_2_y})"

        phi = f"G[0,2] (not ({phi_1})) and (not ({phi_2}))"
        specification = RTAMTDense(phi, {"x_pos" : 0, "y_pos": 1})
        rob = specification.evaluate(traj, time)
        
        return rob

    def get_cost(self, init_point):
        time, traj = self._generate_traj(init_point)   
        traj = traj.T
        # print(traj.shape)
        # print(time.shape)
        phi_1_x = "x_pos <= -1.4 and x_pos >= -1.8"
        phi_1_y = "y_pos <= -1.4 and y_pos >= -1.6"
        phi_1 = f"({phi_1_x}) and ({phi_1_y})"


        phi_2_x = "x_pos <= 4.1 and x_pos >= 3.7"
        phi_2_y = "y_pos <= -1.4 and y_pos >= -1.6"
        phi_2 = f"({phi_2_x}) and ({phi_2_y})"

        phi = f"G[0,2] (not ({phi_1})) and (not ({phi_2}))"
        specification = RTAMTDense(phi, {"x_pos" : 0, "y_pos": 1})
        t = Trace(time, traj.T)
        rob = specification.evaluate(t)#traj, time)

        # phi_unsafe_x = "x_pos <= 0.95 and x_pos >= 0.85"
        # phi_unsafe_y = "y_pos <= 0.95 and y_pos >= 0.85"
        # phi_unsafe = f"G[0,2] (not (({phi_unsafe_x}) and ({phi_unsafe_y})))"
        # specification_unsafe = RTAMTDense(phi_unsafe, {"x_pos": 0, "y_pos": 1})
        # dist_1 = specification_unsafe.evaluate(traj, time)
        dist_2 = self.yellow_polygon_def.distance(Point(init_point))

        # return (max(0,dist_2), rob.value)
        return rob.value
        
    def _set_1_f(self, y, t):
        x1, x2 = y
        derivs = [x1 - x2 + 0.1*t,
                x2*np.cos(2*np.pi*x2) - x1*np.sin(2*np.pi*x1) + 0.1*t]
        return derivs

    def _set_2_f(self, y, t):
        x1, x2 = y
        derivs = [x1,
                -1*x1 + x2]
        return derivs

    # def _generate_traj(self, y0):
    #     assert self.tStop > self.tInc
    #     # print(f"Point Evaluated is {y0}")
    #     assert (y0 in self.green_set) or (y0 in self.yellow_set)
    #     loop_time = self.totalTime/self.tStop
    #     t = np.arange(0., self.tStop, self.tInc)
    #     point_history = []
    #     point_history.append(y0)

    #     time_traj = [0]
    #     if y0 in self.green_set:
    #         green_flag = 1
    #         yellow_flag = 0
    #     elif y0 in self.yellow_set:
    #         green_flag = 0
    #         yellow_flag = 1
    #     time_track = 0
    #     for _ in range(int(loop_time)):
    #         # time_traj.append(t[0])
    #         time_track += t[-1]
    #         time_traj.append(time_track)
            
    #         if (y0 in self.green_set or green_flag == 1) and yellow_flag == 0:
    #             psoln = odeint(self._set_1_f, y0, t)
    #             point_history.append(psoln[-1,:])
    #         elif y0 in self.yellow_set or yellow_flag == 1:
    #             psoln = odeint(self._set_2_f, y0, t)
    #             point_history.append(psoln[-1,:])
    #             # yellow_flag == 1
    #             # green_flag = 0

    #         if y0 in self.green_set:
    #             green_flag = 1
    #             yellow_flag = 0
    #         elif y0 in self.yellow_set:
    #             green_flag = 0
    #             yellow_flag = 1

    #         y0 = psoln[-1,:]
    #     # self.traj_exist = True
    #     return np.array(time_traj), np.array(point_history)

    def _generate_traj(self, y0):
        assert self.tStop > self.tInc
        # print(f"Point Evaluated is {y0}")
        assert (y0 in self.green_set) or (y0 in self.yellow_set)
        loop_time = self.totalTime/self.tStop
        t = np.arange(0., self.tStop, self.tInc)
        point_history = []
        point_history.append(y0)
        switch_history = []
        time_traj = [0]
        if y0 in self.green_set:
            p_flag = 0
            green_flag = 1
            yellow_flag = 0
        elif y0 in self.yellow_set:
            p_flag = 1
            green_flag = 0
            yellow_flag = 1
        time_track = 0
        for _ in range(int(loop_time)):
            # time_traj.append(t[0])
            time_track += t[-1]
            time_traj.append(time_track)
            
            if green_flag == 1 and yellow_flag == 0:
                # if p_flag == 1:
                    # print("g")
                psoln = odeint(self._set_1_f, y0, t)
                point_history.append(psoln[-1,:])
            elif green_flag == 0 and yellow_flag == 1:
                # if p_flag == 1:
                    # print("y")
                psoln = odeint(self._set_2_f, y0, t)
                point_history.append(psoln[-1,:])
                
                
            y0 = psoln[-1,:]
            if y0 in self.green_set and green_flag == 1:
                green_flag = 1
                yellow_flag = 0
            elif y0 in self.yellow_set or yellow_flag == 1:
                green_flag = 0
                yellow_flag = 1

            # print(f"******************************************\nGreen set: {y0 in self.green_set}, green_flag = {green_flag},\nYellow set = {y0 in self.yellow_set}, yellow_flag = {yellow_flag}\n******************************************")
        # self.traj_exist = True
        return np.array(time_traj), np.array(point_history)