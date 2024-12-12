from collections import deque
import random
import torch


class ReplayBuffer(object):
    def __init__(self, capacity: int, important_scale=3):
        # important_scale: 代表重要列表的训练占比多少
        self.common_buffer = deque(maxlen=capacity)
        self.import_buffer = deque(maxlen=int(capacity / important_scale))
        self.now_experience = None
        self.important_scale = 2

    def add_common(self, state, action, reward, next_state, done):
        self.common_buffer.append((state, action, reward, next_state, done))
        self.now_experience = (state, action, reward, next_state, done)

    def add_important(self, state, action, reward, next_state, done):
        self.import_buffer.append((state, action, reward, next_state, done))
        self.now_experience = (state, action, reward, next_state, done)

    def sample(self, batch_size):
        important_size = min(int(batch_size / self.important_scale), self.import_buffer.__len__())
        common_size = batch_size - important_size - 1


        transitions = random.sample(self.import_buffer, important_size)
        if len(self.common_buffer) >= common_size:
            transitions.extend(random.sample(self.common_buffer, common_size))
        else:
            print("Not enough samples in the buffer to draw the desired batch size")

        transitions.append(self.now_experience)

        state, action, reward, next_state, done = zip(*transitions)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        return state, action, reward, next_state, done

    def __len__(self):
        return self.common_buffer.__len__() + self.import_buffer.__len__()