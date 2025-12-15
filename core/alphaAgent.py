import numpy as np

class Bandit:
    def __init__(self, action_size=10):
        self.probs = np.random.rand(action_size)

    def play(self, action):
        return 1 if np.random.rand() < self.probs[action] else 0


class alphaAgent:
  def __init__(self,epsilon,alpha,action_size=10):
    self.epsilon=epsilon
    self.Qs=np.zeros(action_size)
    self.alpha=alpha
      
  def update(self,action,reward):
    self.Qs[action]+=self.alpha*(reward-self.Qs[action])

  def get_action(self):
    if np.random.rand()<self.epsilon:
      return np.random.randint(0,len(self.Qs))
    else:
      return np.argmax(self.Qs)
bandit=Bandit()
agent = alphaAgent(epsilon=0.5,alpha=0.1)

for _ in range(1000):
    action = agent.get_action()
    reward = bandit.play(action)
    agent.update(action, reward)
print(agent.Qs)

[0.25116658 0.61916624 0.99999636 0.71075302 0.72703581 0.46185497
 0.09259164 0.36010937 0.61198197 0.17762815]
