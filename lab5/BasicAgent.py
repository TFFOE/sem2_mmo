import numpy as np
import plotly.express as px
import pandas as pd
import os
import pygame


os.environ['SDL_VIDEODRIVER']='dummy'
pygame.display.set_mode((640,480))


class BasicAgent: #Базовый агент, от которого наследуются стратегии обучения

    # Наименование алгоритма
    ALGO_NAME = 'Base'

    def __init__(self, env, eps=0.1):
        # Среда
        self.env = env
        # Размерности Q-матрицы
        self.nA = env.action_space.n
        self.nS = env.observation_space.n
        #и сама матрица
        self.Q = np.zeros((self.nS, self.nA))
        # Значения коэффициентов
        # Порог выбора случайного действия
        self.eps=eps
        # Награды по эпизодам
        self.episodes_reward = []

    def print_q(self):
        print('Вывод Q-матрицы для алгоритма ', self.ALGO_NAME)
        print(self.Q)

    def get_state(self, state): #Возвращает правильное начальное состояние

        if type(state) is tuple:
            # Если состояние вернулось с виде кортежа, то вернуть только номер состояния
            return state[0]
        else:
            return state

    def greedy(self, state):
        '''
        <<Жадное>> текущее действие
        Возвращает действие, соответствующее максимальному Q-значению
        для состояния state
        '''
        return np.argmax(self.Q[state])

    def make_action(self, state): #Выбор действия агентом

        if np.random.uniform(0,1) < self.eps:

            # Если вероятность меньше eps
            # то выбирается случайное действие
            return self.env.action_space.sample()
        else:
            # иначе действие, соответствующее максимальному Q-значению
            return self.greedy(state)

    def draw_episodes_reward(self):
        # Построение графика наград по эпизодам
        # fig, ax = plt.subplots(figsize = (15,10))
        y = self.episodes_reward
        
        df = pd.DataFrame(data={
            'Номер эпизода': list(range(1, len(y)+1)),
            'Награда': y
        })
        
        fig = px.line(df, x="Номер эпизода", y="Награда", title='Награды по эпизодам', height=400, width=600)
        fig.show()
        # plt.plot(x, y, '-', linewidth=1, color='green')
        # plt.title('')
        # plt.xlabel()
        # plt.ylabel('Награда')
        # plt.show()

    def learn(self):
        '''
        Реализация алгоритма обучения
        '''
        pass