import os
import random
import time

# 라이브러리 import
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# device 설정 (CPU)
device = torch.device('cpu')


# NN의 구조 구현
class Network(nn.Module):
    # 생성자
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()  # nn.Module 사용을 위해서 객체 생성 
        self.input_size = input_size
        self.nb_action = nb_action
        # full connection 1 (input size로부터 30개의 다음 레이어 입력으로)
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, 120)
        # full connection 2 (30개의 입력을 가지고 action 선택)
        self.fc3 = nn.Linear(120, nb_action)
        # self.fc3 = nn.Linear(120, nb_action)

    # forward 함수 (activation func = relu, state 받아서 fc간 연결)
    def forward(self, state):
        # hidden neuron : state를 입력받아 action(Q value)를 반환해줌
        x = F.relu(self.fc1(state))  # nn.Linear(state, 30)
        x = F.relu(self.fc2(x))  # nn.Linear(state, 30)
        q_values = self.fc3(x)  # nn.Linear(30, x)
        return q_values


# experience replay 구현
# 일정 갯수의 시점 경험을 메모리에 저장후 배치사이즈만큼 뽑아서 다음 업데이트 진행
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity  # left capacity : 객체변수, right capacity : 객체 생성 시 매개변수
        self.memory = []

    def push(self, event):  # event : st, st+1, At, Rt
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        # if list = ({1,2,3},{4,5,6}) -> zip(*list) = {(1,4),(2,5),(3,6)}
        samples = zip(*random.sample(self.memory, batch_size))
        # sample 하나를 파이토치 5차원 변수(센서3개+ 2방향)로 바꾸는 함수, samples를 대상으로 map 돌림
        return map(lambda x: Variable(torch.cat(x, 0)).to(device), samples)


# implementing Deep Q learning
# unsqueeze (신경망을 위한 batch 화), squeeze (신경망 사용 후 되돌리기)
# DQN은 large scale env에서 s,a에대한 Q를 근사하기 위한 Q세타를 두고, 그때 가중치 세타를 Deep NN 로 학습하는 방법, exp replay
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action).to(device)  # 모델을 CPU로 설정
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0).to(device)
        self.last_action = 0
        self.last_reward = 0
        self.nb_action = nb_action  # 여기에서 nb_action을 정의

    def select_action(self, state):
        # 현재 state로부터 action을 고를 확률 합 = 1
        probs = F.softmax(
            # self.model(Variable(state, volatile=True).to(device)) * 100, dim=-1
            self.model(Variable(state, volatile=True).to(device)) * 100,
            dim=-1,
        )
        # TODO: 난수 하나 만들고 e보다 작으면 softmax값 반전시켜서 e-greedy 만들기
        # 0과 1 사이의 난수 생성
        # rand_num = random.random()
        # epsilon = 0.1
        # if rand_num < epsilon:
        #     probs = 1 - probs
        #     print("probs inversion")

        # softmax([1,2,3]) = [0.04, 0.11, 0.85] => softmax([1,2,3]*3 = [0,0.02,0.98])
        action = probs.multinomial(num_samples=1)
        # time.sleep(0.1)
        return action.data[0, 0]

    # 배치 정보들을 받아서 loss를 찾아 optimizing함
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = (
            self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        )
        next_outputs = (
            self.model(batch_next_state).detach().max(1)[0]
        )  # s_t+1에서 max q를 가지는 a_t+1을 선택한 경우의 reward 예측 (Q value)
        target = self.gamma * next_outputs + batch_reward.to(device)  # TD target
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)  # retain_variables가 retain_graph로 대체되었습니다
        self.optimizer.step()  # weight update

    # 실제 map에서 t시점에 받아온 정보들로 학습(memory update) 실행, 이번 시점에 고를 action 반환
    def update(self, reward, new_signal):  # 사실상 signal(리스트) = state(torch tensor)
        new_state = torch.Tensor(new_signal).float().unsqueeze(0).to(device)
        # print(new_state)
        # time.sleep(0.4)
        self.memory.push(
            (
                self.last_state,
                new_state,
                torch.LongTensor([int(self.last_action)]).to(device),
                torch.Tensor([self.last_reward]).to(device),
            )
        )  # 모두 torch tensor여야 함 (unsqueeze), 단순 숫자변수 하나인 action은 longTensor로 만듦
        action = self.select_action(new_state)
        if len(self.memory.memory) > 500:
            batch_state, batch_next_state, batch_action, batch_reward = (
                self.memory.sample(500)
            )
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        # 행동을 고르고 sample로 얻은 배치정보들로 MDP S,A,R 업데이트
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.0)

    # 모델의 상태를 저장하는 함수
    def save(self):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            "last_brain.pth",
        )

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done!")
        else:
            print("no checkpoint found...")
