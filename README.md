# **Reinforcement Learning**

##  프로젝트 개요
**딥러닝 밑바닥부터 시작하는 강화학습(4권)**을 기반으로**  
Multi-Armed Bandit부터 바둑 에이전트까지 단계적으로 구현한다.**

---
### agent
1.*확률 epsilon를 따라 새로운 도전을하고 1-epsilon으로 그 도전을 행동한다.*<br>
2.*Qs[action]은 해당 행동을 하였을때의 기대 보상이다.*<br>
3.self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]이 코드는 표본 평균으로 큰 수의 법칙에 다라 시행 횟수가 늘어날수록 진짜 보상값에 가까워진다.

### alphaagent
1.alpha값이 클수록 최근 행동에 대한 가중치가 커짐 즉, 많은 학습을 함<br>
2.alpha값이 작을수록 과거 행동에 대한 영향력이 커짐<br>
3.alpha는 epsilon이 결정한 행동을 얼마나 믿을지의 개념

### gridworld

<img width="450" height="330" alt="image" src="https://github.com/user-attachments/assets/97c9d3e4-e6bf-4417-8800-eaf1717750f1" />

