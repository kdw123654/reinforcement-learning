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

agent = Agent(epsilon=0.5)

for _ in range(1000):
    action = agent.get_action()
    reward = bandit.play(action)
    agent.update(action, reward)
print(agent.Qs)
