# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 15:04:54 2021

Code for decision variables
"""

import numpy as np
import matplotlib.pyplot as plt

class DecisionModel():
    def __init__(self, x = 0, w = 1, wd = 1, ws = 1):
        self.x = x
        self.x1 = x
        self.x2 = x
        
        self.w = w
        
        self.wd = wd
        
        self.ws = ws
        
        self.threshold = 0.2
        
        self.dt = 0.01
        
    def DDMmodel(self, Q):
        
        q1 = Q[0]
        q2 = Q[1]

        dx = self.w * (q1 - q2) * self.dt +  np.random.normal(loc=0.0, scale=0.01)
        
        self.x = self.x + dx
        
        decision = 'None'
        if self.x >= self.threshold:
            decision = 'Right'
        if self.x <= -self.threshold:
            decision = 'Left'
        # if self.x > -self.threshold and self.x < self.threshold:
        #     decision = 'None'
        return self.x, decision
    
    def RacingModel(self, v0, Q):
        
        q1 = Q[0]
        q2 = Q[1]

        dx1 = (v0 + self.w * q1) * self.dt +  np.random.normal(loc=0.0, scale=0.01)
        dx2 = (v0 + self.w * q2) * self.dt +  np.random.normal(loc=0.0, scale=0.01)
        
        self.x1 = self.x1 + dx1
        self.x2 = self.x2 + dx2
        
        if self.x1 >= self.threshold and self.x2 <= self.threshold:
            decision = 'Right'
        if self.x2 >= self.threshold and self.x1 <= self.threshold:
            decision = 'Left'
        if self.x1 <= self.threshold and self.x2 <= self.threshold:
            decision = 'None'
        
        # should write something for conflict resolution
        if self.x1 >= self.threshold and self.x2 >= self.threshold:
            flag = np.random.randint(2)
            if flag == 0:
                decision = 'Right'
            else:
                decision = 'Left'
            
        return self.x1, self.x2, decision
    
    def AdvRacingModel(self, v0, Q):
        
        q1 = Q[0]
        q2 = Q[1]

        dx1 = (v0 + self.wd * (q1 - q2) + self.ws * (q1 + q2)) * self.dt +  np.random.normal(loc=0.0, scale=0.01)
        dx2 = (v0 + self.wd * (q2 - q1) + self.ws * (q1 + q2)) * self.dt +  np.random.normal(loc=0.0, scale=0.01)
        
        self.x1 = self.x1 + dx1
        self.x2 = self.x2 + dx2
        
        if self.x1 >= self.threshold and self.x2 <= self.threshold:
            decision = 'Right'
        if self.x2 >= self.threshold and self.x1 <= self.threshold:
            decision = 'Left'
        if self.x1 <= self.threshold and self.x2 <= self.threshold:
            decision = 'None'
        
        # should write something for conflict resolution
        if self.x1 >= self.threshold and self.x2 >= self.threshold:
            flag = np.random.randint(2)
            if flag == 0:
                decision = 'Right'
            else:
                decision = 'Left'
            
        return self.x1, self.x2, decision
    
    def updateQvals(self, Q, reward, alpha, decision):
        if decision == 'Right':
            Q[0] = Q[0] + alpha * (reward - Q[0])
        if decision == 'Left':
            Q[1] = Q[1] + alpha * (reward - Q[1])
        return Q
    
def TaskProcess(decision):
    if decision == 'Left':
        prob_status = np.random.uniform()
        if prob_status <= 0.8:
            reward = 100
        else:
            reward = 0
    if decision == 'Right':
        reward = 0
    
    if decision == 'None':
        reward = 0   
    return reward
    
if __name__ == '__main__':
    # This section gives task
    
    
    task_dur = 2.5 # time in seconds
    ds = 1 #0.01 # sampling time
    time_steps = np.linspace(0, int(task_dur/ds - 1), int(task_dur/ds))
    total_trials = 1
    trials = 0
    
    Q_init = np.zeros((2,1))
    Q_cur = Q_init
    
    num_rights = 0
    num_lefts = 0
    while trials <= total_trials:
        x = 0
        w = 1
        wd = 1
        ws = 1
        alpha = 0.001
        #Q_init = np.zeros((2,1))
        #Q_cur = Q_init
        decision_ins = DecisionModel(x, w, wd, ws)
        total_x = np.zeros((1,2))
        decision = 'None'
        print(total_x)

        for t in time_steps:
            
            # Take decision
            x1, x2, decision = decision_ins.AdvRacingModel(x, Q_cur)
            
            # Obtain reward
            reward = TaskProcess(decision)
            
            total_x = np.append(total_x, [decision_ins.x1, decision_ins.x2])
            print(total_x)
            if decision == 'Right' or decision == 'Left':
                # Update the Q-values with reward
                Q_cur = decision_ins.updateQvals(Q_cur, reward, alpha, decision)
                #Q_cur = decision_ins.updateQvals(Q_cur, reward, alpha)
                
                break

        # print(decision_ins.x)
        # print(decision)
        # print(reward)
        # print(Q_cur)
        # print(total_x[10])
        if decision == 'Right':
            num_rights += 1
        else:
            num_lefts += 1

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(total_x[0], '--b')
        plt.plot(total_x[1], '--r')
        plt.plot(0.2 + total_x * 0, '--g')
        plt.plot(-0.2 + total_x * 0, '--g')
        plt.pause(0.1)

        # Update the Q-values with reward
        
            
        
        
        trials += 1