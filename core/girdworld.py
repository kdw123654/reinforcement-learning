import numpy as np
from collections import defaultdict

# ==========================================
# 1. GridWorld 환경 클래스
# ==========================================
class gridworld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        self.reward_map = np.array([[0, 0, 0, 1.0],
                                    [0, None, 0, -1.0],
                                    [0, 0, 0, 0]])
        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    def height(self): return len(self.reward_map)
    def width(self): return len(self.reward_map[0])
    def actions(self): return self.action_space
    def states(self):
        for h in range(self.height()):
            for w in range(self.width()):
                yield (h, w)

    def next_state(self, state, action):
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state
        
        if nx < 0 or nx >= self.width() or ny < 0 or ny >= self.height():
            next_state = state
        elif next_state == self.wall_state:
            next_state = state
            
        return next_state

    def reward(self, state, action, next_state):
        return self.reward_map[next_state[0], next_state[1]]

    def render_v(self, V):
        for h in range(self.height()):
            for w in range(self.width()):
                state = (h, w)
                if state == self.wall_state:
                    print("  ###  ", end="")
                else:
                    print(f"{V[state]:6.2f}", end=" ")
            print()
        print()


def argmax(d):
    return max(d, key=d.get)


def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        if state == env.wall_state:
            continue
            
        action_probs = pi[state]
        new_V = 0
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V

def policy_eval(pi, V, env, gamma, threshold=0.001):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)
        
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
        if delta < threshold:
            break
    return V

def greedy_policy(V, env, gamma):
    pi = {}
    for state in env.states():
        if state == env.goal_state or state == env.wall_state:
            pi[state] = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
            continue
            
        action_values = {}
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value
            
        max_action = argmax(action_values)
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi

def policy_iter(env, gamma, threshold=0.001):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)
        
        if new_pi == pi:
            break
        pi = new_pi
    return V, pi  


def value_iter_onestep(V, env, gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        if state == env.wall_state: 
            continue
            
        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)
            
        V[state] = max(action_values) 
    return V

def value_iter(V, env, gamma, threshold=0.001):
    while True:
        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)
        
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
        
        if delta < threshold:
            print("Value Iteration Converged!")
            break
    return V

# ==========================================
# 5. 실행부 (Main)
# ==========================================
if __name__ == "__main__":
    env = gridworld()
    gamma = 0.9
    
    # 1. 정책 반복법 실행
    print("=== Policy Iteration Result ===")
    V_pi, pi = policy_iter(env, gamma)
    env.render_v(V_pi)

    # 2. 가치 반복법 실행
    print("\n=== Value Iteration Result ===")
    V_vi = defaultdict(lambda: 0)
    final_V = value_iter(V_vi, env, gamma)
    env.render_v(final_V)


=== Policy Iteration Result ===
  0.81   0.90   1.00   0.00 
  0.73   ###    0.90   1.00 
  0.66   0.73   0.81   0.73 


=== Value Iteration Result ===
Value Iteration Converged!
  0.81   0.90   1.00   0.00 
  0.73   ###    0.90   1.00 
  0.66   0.73   0.81   0.73 
