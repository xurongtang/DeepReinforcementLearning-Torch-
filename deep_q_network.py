import os
import cv2
import time
import random
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 导入 Flappy Bird 环境
import game.wrapped_flappy_bird as game

import warnings
warnings.filterwarnings('ignore')


# ---------------------- 超参数 ----------------------
GAME = 'FlappyBird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 1000
EXPLORE = 200000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-6


# ---------------------- DQN 网络 ----------------------
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, ACTIONS)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # [B, 32, 20, 20]
        x = F.relu(self.conv2(x))   # [B, 64, 9, 9]
        x = F.relu(self.conv3(x))   # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ---------------------- 图像预处理 ----------------------
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (80, 80, 1))


# ---------------------- 训练主函数 ----------------------
def train():
    writer = SummaryWriter(log_dir=f"runs/{GAME}_dqn")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    game_state = game.GameState()
    D = deque()

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, _, _ = game_state.frame_step(do_nothing)
    x_t = preprocess(x_t)
    s_t = np.concatenate([x_t] * 4, axis=2)

    epsilon = INITIAL_EPSILON
    t = 0

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    while True:
        s_tensor = torch.from_numpy(s_t.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        output = model(s_tensor)
        a_t = np.zeros([ACTIONS])
        action_index = 0

        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                action_index = torch.argmax(output).item()
                a_t[action_index] = 1
        else:
            a_t[0] = 1

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        x_t1, reward, terminal = game_state.frame_step(a_t)
        x_t1 = preprocess(x_t1)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        D.append((s_t, a_t, reward, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE:
            minibatch = random.sample(D, BATCH)
            s_batch = torch.tensor(np.array([d[0].transpose(2, 0, 1) for d in minibatch]), dtype=torch.float32).to(device)
            a_batch = torch.tensor(np.array([d[1] for d in minibatch]), dtype=torch.float32).to(device)
            r_batch = torch.tensor(np.array([d[2] for d in minibatch]), dtype=torch.float32).to(device)
            s1_batch = torch.tensor(np.array([d[3].transpose(2, 0, 1) for d in minibatch]), dtype=torch.float32).to(device)

            q_values = model(s_batch)
            q_action = torch.sum(q_values * a_batch, dim=1)

            q_next = model(s1_batch)
            y = []
            for i in range(BATCH):
                if minibatch[i][4]:
                    y.append(r_batch[i])
                else:
                    y.append(r_batch[i] + GAMMA * torch.max(q_next[i]))
            y = torch.stack(y)

            loss = F.mse_loss(q_action, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 日志记录
            writer.add_scalar("Loss", loss.item(), t)
            writer.add_scalar("Q_max", output.max().item(), t)
            writer.add_scalar("Reward", reward, t)
            writer.add_scalar("Epsilon", epsilon, t)

        s_t = s_t1
        t += 1

        if t % 10000 == 0:
            torch.save(model.state_dict(), f"saved_models/{GAME}_dqn_{t}.pth")

        print(f"[{t}] STATE: {'observe' if t <= OBSERVE else 'train'} | EPSILON: {epsilon:.5f} | ACTION: {action_index} | REWARD: {reward} | Q_MAX: {output.max().item():.5f}")


if __name__ == "__main__":
    train()
