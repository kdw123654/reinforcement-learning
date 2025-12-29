# **Reinforcement Learning**

##  프로젝트 개요
**딥러닝 밑바닥부터 시작하는 강화학습(4권)**을 기반으로**  
Multi-Armed Bandit부터 바둑 에이전트까지 단계적으로 구현한다.**

---
### agent
1.*확률 epsilon를 따라 새로운 도전을하고 1-epsilon으로 그 도전을 행동한다.*<br>
2.*Qs[action]은 해당 행동을 하였을때의 기대 보상이다.*<br>
3.self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]이 코드는 표본 평균으로 큰 수의 법칙에 다라 시행 횟수가 늘어날수록 진짜 보상값에 가까워진다

### alphaagent
1.alpha값이 클수록 최근 행동에 대한 가중치가 커짐 즉, 많은 학습을 한다<br>
2.alpha값이 작을수록 과거 행동에 대한 영향력이 커진다<br>
3.alpha는 epsilon이 결정한 행동을 얼마나 믿을지의 개념이다

### gridworld

<img width="450" height="330" alt="image" src="https://github.com/user-attachments/assets/97c9d3e4-e6bf-4417-8800-eaf1717750f1" />

1.벨만 방정식을 이용한 정책 평가 이용>> 우리가 설정한 기준대로 로봇이 움직이면 평균적으로 얼마의 값을 가지는지에 대한 방법이다<br>
2.벨만 방정식을 이용한 가치 반복 이용>> 무조건 가장 큰 값을 따라갔을때 얼마의 값을 가질 수 있는지에 대한 방법이다<br>
3.girdworld는 우리가 미리 틀을 정해주고 어느정도 기준을 세워줘야하지만 현실 문제는 그렇지 못함 따라서 이를 바탕으로 Q-Learning과같은 더 복잡한 알고리즘이 필요하다
