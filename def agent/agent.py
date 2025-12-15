import numpy as np

class Bandit:
    def __init__(self, action_size=10):
        self.probs = np.random.rand(action_size)

    def play(self, action):
        return 1 if np.random.rand() < self.probs[action] else 0


class Agent:
  def __init__(self,epsilon,action_size=10):
    self.epsilon=epsilon
    self.Qs=np.zeros(action_size)
    self.ns=np.zeros(action_size)

  def update(self,action,reward):
    self.ns[action]+=1
    self.Qs[action]+=(reward-self.Qs[action])/self.ns[action]

  def get_action(self):
    if np.random.rand()<self.epsilon:
      return np.random.randint(0,len(self.Qs))
    else:
      return np.argmax(self.Qs)
bandit=Bandit
agent = Agent(epsilon=0.5)

for _ in range(1000):
    action = agent.get_action()
    reward = bandit.play(action)
    agent.update(action, reward)
print(agent.Qs)


[0.9001996  0.59183673 0.16981132 0.72       0.46938776 0.82608696
 0.37931034 0.85185185 0.30952381 0.82666667]
