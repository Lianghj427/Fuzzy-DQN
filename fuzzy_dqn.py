import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.414)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.LongTensor(np.array(action)).to(device),
            torch.FloatTensor(np.array(reward)).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(np.array(done)).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

class FuzzyMembership:
    def __init__(self):

        self.angle_small = (-0.1, 0, 0.1)
        self.angle_medium_neg = (-0.25, -0.15, -0.05)
        self.angle_medium_pos = (0.05, 0.15, 0.25)
        self.angle_large_neg = (-0.5, -0.3, -0.1)
        self.angle_large_pos = (0.1, 0.3, 0.5)
        
        self.velocity_small = (-0.5, 0, 0.5)
        self.velocity_medium_neg = (-1.5, -1.0, -0.5)
        self.velocity_medium_pos = (0.5, 1.0, 1.5)
        self.velocity_large_neg = (-3.0, -2.0, -1.0)
        self.velocity_large_pos = (1.0, 2.0, 3.0)
    
    def triangle_mf(self, x, params):

        a, b, c = params
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)
    
    def get_angle_memberships(self, angle):

        memberships = {
            "small": self.triangle_mf(angle, self.angle_small),
            "medium_neg": self.triangle_mf(angle, self.angle_medium_neg),
            "medium_pos": self.triangle_mf(angle, self.angle_medium_pos),
            "large_neg": self.triangle_mf(angle, self.angle_large_neg),
            "large_pos": self.triangle_mf(angle, self.angle_large_pos)
        }
        return memberships
    
    def get_velocity_memberships(self, velocity):

        memberships = {
            "small": self.triangle_mf(velocity, self.velocity_small),
            "medium_neg": self.triangle_mf(velocity, self.velocity_medium_neg),
            "medium_pos": self.triangle_mf(velocity, self.velocity_medium_pos),
            "large_neg": self.triangle_mf(velocity, self.velocity_large_neg),
            "large_pos": self.triangle_mf(velocity, self.velocity_large_pos)
        }
        return memberships

class FuzzyDQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003)

        self.memory = ReplayBuffer(20000)

        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.005

        self.fuzzy = FuzzyMembership()

        self.fuzzy_weight = 0.3

        self.steps_done = 0

        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
    
    def get_fuzzy_action_preference(self, state):

        cart_pos = state[0]
        cart_vel = state[1]
        pole_angle = state[2]
        pole_vel = state[3]

        angle_memberships = self.fuzzy.get_angle_memberships(pole_angle)
        velocity_memberships = self.fuzzy.get_velocity_memberships(pole_vel)
        
        rule1 = min(angle_memberships["medium_neg"] + angle_memberships["large_neg"], 
                   velocity_memberships["medium_neg"] + velocity_memberships["large_neg"])
        
        rule2 = min(angle_memberships["medium_pos"] + angle_memberships["large_pos"], 
                   velocity_memberships["medium_pos"] + velocity_memberships["large_pos"])
        
        rule3_left = angle_memberships["large_neg"] * velocity_memberships["medium_pos"]
        rule3_right = angle_memberships["medium_neg"] * velocity_memberships["medium_pos"]
        
        rule4_left = angle_memberships["medium_pos"] * velocity_memberships["medium_neg"]
        rule4_right = angle_memberships["large_pos"] * velocity_memberships["medium_neg"]
        
        rule5 = min(angle_memberships["small"], velocity_memberships["small"])
        
        left_preference = rule1 + rule3_left + rule4_left
        right_preference = rule2 + rule3_right + rule4_right
        
        if rule5 > 0.7:
            return 0
        
        return right_preference - left_preference
    
    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        state = state.to(self.device)
        
        sample = random.random()
        eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
                        np.exp(-1. * self.steps_done / 1000)
        self.steps_done += 1
        
        if sample < eps_threshold:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():

            q_values = self.policy_net(state)
            
            fuzzy_pref = self.get_fuzzy_action_preference(state.cpu().numpy())
            
            fuzzy_bonus = torch.zeros_like(q_values)
            if fuzzy_pref > 0:
                fuzzy_bonus[0, 1] = fuzzy_pref * self.fuzzy_weight
            elif fuzzy_pref < 0:
                fuzzy_bonus[0, 0] = -fuzzy_pref * self.fuzzy_weight
            
            final_q = q_values + fuzzy_bonus
            
            return final_q.max(1)[1].item()
    
    def update_model(self):
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():

            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)

            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
        
        if self.steps_done % 1000 == 0 and self.fuzzy_weight > 0.05:
            self.fuzzy_weight *= 0.95
        
        return loss.item()
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'fuzzy_weight': self.fuzzy_weight
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.fuzzy_weight = checkpoint.get('fuzzy_weight', self.fuzzy_weight)