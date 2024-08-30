import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pygame


# Q-Table 에이전트 정의
class QTableAgent:
    def __init__(self, state_bins, action_size, env):
        self.state_bins = state_bins
        self.action_size = action_size
        self.q_table = np.zeros(state_bins + (action_size,))  # 상태-행동에 대한 Q 테이블 초기화
        self.gamma = 0.95  # 할인율
        self.epsilon = 1.0  # 탐험 비율 초기값
        self.epsilon_min = 0.01  # 탐험 비율 최소값
        self.epsilon_decay = 0.995  # 탐험 비율 감소율
        self.learning_rate = 0.8  # 학습률
        self.env = env  # 환경 객체 저장

    def discretize_state(self, state):
        """상태를 이산화하여 Q 테이블에 사용."""
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_bins = [np.linspace(low, high, num - 1) for low, high, num in zip(env_low, env_high, self.state_bins)]
        state_index = tuple(
            np.digitize(state[i], env_bins[i]) for i in range(len(state))
        )
        return state_index

    def act(self, state):
        """탐험 또는 최대 Q값 기반 행동 선택."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # 무작위 행동
        state_index = self.discretize_state(state)  # 상태 이산화
        return np.argmax(self.q_table[state_index])  # Q 테이블에서 최대 Q값을 가지는 행동 선택

    def learn(self, state, action, reward, next_state, done):
        """Q 테이블 갱신."""
        state_index = self.discretize_state(state)
        next_state_index = self.discretize_state(next_state)
        target = reward + (0 if done else self.gamma * np.max(self.q_table[next_state_index]))
        self.q_table[state_index][action] = (1 - self.learning_rate) * self.q_table[state_index][
            action] + self.learning_rate * target

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # 탐험 비율 감소


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
        state = np.reshape(state, [state_size])

        # Pygame 화면에 CartPole 환경을 시각화
        screen.fill((255, 255, 255))
        img = env.render()
        img = np.rot90(img)
        img = pygame.surfarray.make_surface(img)
        screen.blit(img, (0, 0))
        pygame.display.flip()

        clock.tick(30)

    pygame.quit()
    env.close()


# 학습 및 시각화
def train_q_table(episodes):
    env = gym.make("CartPole-v1")
    state_bins = (6, 6, 6, 6)  # 각 상태에 대한 이산화 구간 설정
    action_size = env.action_space.n  # 가능한 행동 수
    agent = QTableAgent(state_bins, action_size, env)  # 환경을 포함하여 QTableAgent 생성
    scores = []

    plt.ion()

    for e in range(episodes):
        state, _ = env.reset()
        for time in range(500):
            action = agent.act(state)  # 행동 선택
            next_state, reward, done, _, _ = env.step(action)  # 환경에서 행동 수행
            reward = reward if not done else -10  # 에피소드 종료 시 보상을 -10으로 설정
            agent.learn(state, action, reward, next_state, done)  # Q 테이블 갱신
            state = next_state  # 다음 상태로 전이
            if done:
                scores.append(time)
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                run_trained_agent(agent)
                break

        # 진행 상황 시각화
        if e % 10 == 0:
            plt.clf()
            plt.plot(scores)
            plt.xlabel('Episode')
            plt.ylabel('Duration')
            plt.title('Training Progress')
            plt.pause(0.001)

    plt.ioff()
    plt.show()

    return agent, scores


episodes = 100
agent, scores = train_q_table(episodes)
