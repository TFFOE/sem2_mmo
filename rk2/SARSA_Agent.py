import os
import pygame
from BasicAgent import BasicAgent


os.environ['SDL_VIDEODRIVER']='dummy'
pygame.display.set_mode((640,480))


class SARSA_Agent(BasicAgent):
    ALGO_NAME = 'SARSA'

    def __init__(self, env, eps=0.4, lr=0.1, gamma=0.98):
        super().__init__(env, eps)
        self.lr=lr
        self.gamma = gamma
        self.eps_decay=0.00005
        self.eps_threshold=0.01

    def learn(self, num_episodes=20000):
        self.episodes_reward = []
        for ep in list(range(num_episodes)):
            state = self.get_state(self.env.reset())
            done = False
            truncated = False
            tot_rew = 0

            if self.eps > self.eps_threshold:
                self.eps -= self.eps_decay

            action = self.make_action(state)

            while not (done or truncated):
                next_state, rew, done, truncated, _ = self.env.step(action)

                next_action = self.make_action(next_state)

                self.Q[state][action] = self.Q[state][action] + self.lr * \
                    (rew + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])

                state = next_state
                action = next_action
                tot_rew += rew
                if (done or truncated):
                    self.episodes_reward.append(tot_rew)