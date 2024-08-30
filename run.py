import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import pygame

# Q-Network 정의
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# DQN 에이전트 정의
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state)
            target_f = target_f.clone().detach()  # 기존 텐서와 연결을 끊음
            target_f[0][action] = target  # action에 대한 값 갱신
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def run_trained_agent(agent):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state, _ = env.reset()
    state_size = env.observation_space.shape[0]

    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((600, 400))

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
        state = np.reshape(state, [1, state_size])

        # Pygame 화면에 CartPole 환경을 시각화
        screen.fill((255, 255, 255))
        img = env.render()
        img = np.rot90(img)  # 이미지 회전 (필요시 조정)
        img = pygame.surfarray.make_surface(img)
        screen.blit(img, (0, 0))
        pygame.display.flip()

        clock.tick(30)  # FPS 제한

    pygame.quit()
    env.close()


# 학습 및 시각화
def train_dqn(episodes):
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    done = False
    scores = []

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot(scores)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Duration')
    ax.set_title('Training Progress')

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                scores.append(time)
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                run_trained_agent(agent)
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # Update plot after each episode
        line.set_ydata(scores)
        line.set_xdata(range(len(scores)))
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

    plt.ioff()
    plt.show()

    return agent, scores


episodes = 100
agent, scores = train_dqn(episodes)