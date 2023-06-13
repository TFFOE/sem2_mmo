import numpy as np
import plotly.express as px
import pandas as pd
import os
import pygame


os.environ['SDL_VIDEODRIVER']='dummy'
pygame.display.set_mode((640,480))


class BasicAgent:

    ALGO_NAME = 'Base'

    def __init__(self, env, eps=0.1):
        self.env = env
        self.nA = env.action_space.n
        self.nS = env.observation_space.n
        self.Q = np.zeros((self.nS, self.nA))
        self.eps=eps
        self.episodes_reward = []
        self.all_reward = 0

    def print_q(self):
        self.all_reward = np.sum(self.Q)

    def get_state(self, state):

        if type(state) is tuple:
            return state[0]
        else:
            return state

    def greedy(self, state):
        return np.argmax(self.Q[state])

    def make_action(self, state):

        if np.random.uniform(0,1) < self.eps:
            return self.env.action_space.sample()
        else:
            return self.greedy(state)

    def draw_episodes_reward(self):
        y = self.episodes_reward
        
        df = pd.DataFrame(data={
            'Номер эпизода': list(range(1, len(y)+1)),
            'Награда': y
        })
        
        fig = px.line(df, x="Номер эпизода", y="Награда", title='Награды по эпизодам', height=400, width=600)
        fig.show()

    def learn(self):
        pass