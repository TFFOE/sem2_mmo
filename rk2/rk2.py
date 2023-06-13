import gymnasium as gym
import os
import pygame
from tabulate import tabulate
import time
import numpy as np
from tqdm import tqdm
from SARSA_Agent import SARSA_Agent


os.environ['SDL_VIDEODRIVER']='dummy'
pygame.display.set_mode((640,480))


def run_sarsa():
    all_rewards = []
    parameters = []
    
    lr_list = np.linspace(0.0005, 0.005, num=5)
    gamma_list = np.linspace(0.9, 1, num=5)
    eps_list = np.linspace(0.05, 0.9, num=9)
    
    env = gym.make('Taxi-v3')
    for lr in tqdm(lr_list, bar_format=' {l_bar}{bar:20}{r_bar}{bar:-10b}', colour='CYAN'):
        for gamma in gamma_list:
            for ep in eps_list:
                agent = SARSA_Agent(env, lr=lr, gamma=gamma, eps=ep)
                agent.learn(100)
                agent.print_q()
                all_rewards.append(agent.all_reward)
                parameters.append([lr, gamma, ep])

    return all_rewards, parameters
    

def main():
    all_rewards, parameters = run_sarsa()

    print(tabulate(
        {
            'Максимальная награда:' : [np.max(all_rewards)],
            'Значения гиперпараметров' : parameters[np.argmax(np.max(all_rewards))]
        }, 
        headers='keys', 
        tablefmt='psql'))
    print(f"Закончено за {time.process_time():.3f}")


if __name__ == '__main__':
    main()
