import gymnasium as gym
import os
import pygame
import asyncio


from DoubleQLearning_Agent import DoubleQLearning_Agent
from QLearning_Agent import  QLearning_Agent
from SARSA_Agent import SARSA_Agent


os.environ['SDL_VIDEODRIVER']='dummy'
pygame.display.set_mode((640,480))


def play_agent(agent):
    '''
    Проигрывание сессии для обученного агента
    '''
    env2 = gym.make('Taxi-v3', render_mode='human')
    state = env2.reset()[0]
    done = False
    while not done:
        action = agent.greedy(state)
        next_state, reward, terminated, truncated, _ = env2.step(action)
        env2.render()
        state = next_state
        if terminated or truncated:
            done = True


def run_sarsa():
    env = gym.make('Taxi-v3')
    agent = SARSA_Agent(env)
    agent.learn()
    agent.print_q()
    agent.draw_episodes_reward()
    play_agent(agent)


def run_q_learning():
    env = gym.make('Taxi-v3')
    agent = QLearning_Agent(env)
    agent.learn()
    agent.print_q()
    agent.draw_episodes_reward()
    play_agent(agent)


def run_double_q_learning():
    env = gym.make('Taxi-v3')
    agent = DoubleQLearning_Agent(env)
    agent.learn()
    agent.print_q()
    agent.draw_episodes_reward()
    play_agent(agent)
    

async def main():
    treads = [asyncio.to_thread(run_q_learning), asyncio.to_thread(run_sarsa), asyncio.to_thread(run_double_q_learning)]
    
    tasks = [asyncio.create_task(tread) for tread in treads]
    
    await asyncio.gather(*tasks)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()