# **Reinforcement Learning**

##  프로젝트 개요
**딥러닝 밑바닥부터 시작하는 강화학습(4권)**을 기반으로**  
Multi-Armed Bandit부터 바둑 에이전트까지 단계적으로 구현한다.**

---
### agent함수
1.*확률 epsilon를 따라 새로운 도전을하고 1-epsilon으로 그 도전을 행동한다.*<br>
2.*Qs[action]은 해당 행동을 하였을때의 기대 보상이다.*<br>
3.self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]이 코드는 표본 평균으로 큰 수의 법칙에 다라 시행 횟수가 늘어날수록 진짜 보상값에 가까워진다.
