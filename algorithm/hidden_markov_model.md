# 🚀 학습 키워드

핵심 키워드

    1.	순차 데이터 (Sequential Data)
      -	특정 순서를 가지는 데이터 예: 쇼핑몰 구매 목록, 영화 시청 목록, 유전자 서열, 주식 등
    2.	예측 문제 (Forecasting Problem)
      -	시계열 데이터를 사용하여 미래를 예측하는 문제
    3.	마르코프 과정 (Markov Process)
      -	미래의 상태가 현재 상태에만 의존하는 확률 과정
    4.	상태 전이 행렬 (State Transition Matrix)
      -	각 상태에서 다른 상태로의 전환 확률을 나타내는 행렬
    5.	은닉 마르코프 모델 (Hidden Markov Model, HMM)
      -	숨겨진 상태에 의해 관측 가능한 이벤트가 발생한다고 가정하는 통계적 모델

주요 문제

    1.	평가 문제 (Evaluation Problem)
      -	주어진 HMM과 관측된 시퀀스의 확률을 계산하는 문제, 전방 알고리즘(Forward Algorithm) 사용
    2.	디코딩 문제 (Decoding Problem)
      -	주어진 HMM과 관측된 시퀀스에 대해 가장 그럴듯한 숨겨진 상태 시퀀스를 결정하는 문제, 비테르비 알고리즘(Viterbi Algorithm) 사용

알고리즘

    -	전방 알고리즘 (Forward Algorithm)
      -	HMM에서 관측된 시퀀스가 나타날 확률을 계산
    -	비테르비 알고리즘 (Viterbi Algorithm)
      -	주어진 관측 시퀀스에 대해 가장 가능성 높은 숨겨진 상태 시퀀스를 찾음

---

# 📝 새로 배운 개념

## Sequential Data

- 특정 순서를 가지는 데이터
- ex : 쇼핑몰 구매 목록, 영화 시청 목록, 유전자 서열, 주식..등

## Forecasting Problem

- 시계열 데이터를 이용하여 미래를 예측하는 문제
- N discrete states 중 하나의 상태를 갖는 시스템을 고려 -> $q_t \in$ { ${S_1, S_2, ..., S_N}$ }
- 정의된 상태는 랜덤으로 변하는 stochastic systems (확률론적인 시스템) 이라고 가정
- 시스템의 상태는 관측되지 않는 hidden state이며, 관측되는 것은 observation이다.
  - 이때 joint distribution(결합 확률 분포)는 거의 계산 불가능하다 -> $p(q_0, q_1, ..., q_T) = P(q_0)P(q_1|q_0)P(q_2|q_1q_0)P(q_3|q_2q_1q_0)...$

```
결합 확률 분포 : 여러 확률 변수가 취할 수 있는 모든 값의 조합에 대한 확률을 설명하는 확률 분포
- ex) P(A, B) = P(A)P(B|A) = P(B)P(A|B)
- 두개 이상의 확률 변수가 동시에 어떤 값들을 취할 확률을 나타냄
```

- 특정 상태의 확률 값은 `이전 상태의 확률 값`을 알면 구할 수 있다.
- 과거의 모든 상태를 구하는 것은 거의 불가능하다 (ex: 빅뱅 -> 조선 -> 10일전 등,,)

## Markov Property (Markov Assumption)

- 다음 미래의 상태는 오직 현재 상태에 영향만 받는다(가정)
  - $P(q_{t+1}|q_t,...,q_{0}) = P(q_{t+1}|q_t)$
- 현재 상태가 주어지고 **이전 과거의 상태는 고려하지 않는다.**
- 바로 직전의 정보가 충분히 과거의 정보를 담고 있다고 가정한다.
- 현재 상태가 미래를 예측하는 데 충분한 정보를 담고 있다고 가정한다.

## Markov process

- **랜덤 프로세스**이다,
  - 확률적인 시스템의 일종으로 미래의 상태가 오직 현재 상태에만 의존하면 과거의 상태나 어떻게 현재 상태에 이르렀는지는 영향을 미치지 않는 과정을 말함
- 확률적인 행동(Stochastic behavior)을 표현, 연속적인 행동의 변화를 관찰
  - a finite set of N states, $S =$ { ${S_1, S_2, ..., S_N}$ }
  - a state transition probability, $P = {p_{ij}}_{M * M}, 1 \leq i, j \leq M$
  - an initial state distribution, $\pi = $ { $\pi_i$ }

```
전이 확률 (Transition Probability) : 한 상태에서 다른 상태로 이동할 확률
- 이 확률은 상태간의 전이를 결정 짓는 핵심 요소
```

## State Transition Matrix

- Markov state s와 성공 상태 s'에 대해, 상태 변화 확률은 다음과 같이 정의
  - $P_{ss'} = P(S_{t+1} = s' | S_t = s)$

```
- Markov state : 마르코프 과정에서 특정 시간 t에서 시스템이 취할 수 있는 상태를 나타냄
    - 이 상태에서 시스템은 다음 상태로 이동할 확률이 정의
- 성공 상태 s' : 주어진 상태 s에서 다음 시간 t+1에서 시스템이 취할 수 있는 상태 중 하나
    - 현재 상태 s만이 미래 상태 s'에 영향을 준다는 것이며, 과거의 어떤 상태도 s'에 영향을 주지 않는다.
```

- 상태변화 행렬 P는 모든 상태 s로부터 다음 상태로 s'로 변할(전환될) 확률을 정의
- 각 행과 열이 마르코프 체인의 상태를 나타냄
  - 각 행은 현재 상태에서 출발하여 다른 상태로 이동할 확률
  - 각 열은 도착 상태를 나타냄
    $$ P = \begin{bmatrix} p*{11} & p*{12} & ... & p*{1n} \\ p*{21} & p*{22} & ... & p*{2n} \\ ... & ... & ... & ... \\ p*{n1} & p*{n2} & ... & p\_{nn} \end{bmatrix}$$
- 특징
  - `각 행의 합은 1 이다` : 어떤 상태에서 다른 모든 상태로의 전환 확률의 합이 항상 1이 되어야 한다는 것을 의미, 어떤 상태에서 무조건 다른 상태로 이동해야 하기 때문
    - $\sum_{j \in s}{} p_{ij} = 1$
  - `비음수` : 모든 전환 확률은 음수가 아닌 값이어야 한다.

## Hidden Markov Model (HMM)

- 관측 가능한 이벤트들이 내부적인 숨겨진 상태에 의해 생성된다고 가정하는 통계적 모델
  - 간단하게 말해 **같은 시간(동시에)** 에 발생한 두 종류의 state sequence 각각의 특성과 그들의 관계를 모델링

```
state sequence : 시간에 따라 나타나는 일련의 상태들을 의미
- 각 시간 단계에서 상태가 바뀔 수 있는 확률적 프로세스, 다음 상태는 오직 현재 상태에만 의존하는 '무기억성' 속성을 가짐
```

### HMM 구성요소

1. Hidden States Sequence (숨겨진 상태 시퀀스)
   - 모델의 내부 상태를 나타내며, 직접 관측할 수 없다.
   - $(S_1, S_2, ..., S_{t-1}, S_t)$
   - `Markov assumption`을 따름 -> 순차적 특성을 반영
2. Observable States Sequence (관측 가능한 상태 시퀀스)
   - 숨겨진 상태에 의해 영향을 받는 관측 가능한 이벤트들의 시퀀스
   - $(S_{1}^{'}, S_{2}^{'}, ..., S_{t-1}^{'}, S_t^{'})$
   - 순차적 특성을 반영하는 Hidden state에 종속

### HMM: Parameters

- Parameters of a Hidden Markov Model : $\lambda = (A, B, \pi)$

1. $A(a_{ij})$ : 상태 전이 확률 행렬 (State Transition Probability Matrix)
   - 한 숨겨진 상태 i에서 다른 숨겨진 상태 j로 이동할 확률을 정의
   - $a_{ij} = P(S_{t+1} = j | S_t = i)$ , $1 \leq i, j \leq n$
   - $\sum*{j=1}^{n} a*{ij} = 1$
2. $B(b_{jk})$ : 방출 확률 행렬 (Emission Probability Matrix)
   - 특정 숨겨진 상태 j에서 각 관측 가능한 상태 k가 나타날 확률을 정의 (은닉 상태 bj에서 관측치가 vk가 도출될 확률)
   - $b_j(vk) = P(o_t = v_k | q_t = s_j)$, $1 \leq j \leq n, 1 \leq k \leq m$
   - $\sum\_{j=1}^{n} b_j(v_k) = 1$
3. $\pi = (\pi _ {i})$ : 초기 상태 확률 (Initial State Probability)
   - 시퀀스의 시작에서 각 숨겨진 상태를 갖게 될 확률을 정의
   - $\pi$ -> HMM을 가동 시킬 때 어느 상태에서 시작할 지 결정
   - $\pi_i -> s_i$에서 시작할 확률
   - $\sum\_{i=1}^{n} \pi_i = 1$

## HMM Problems

### Evaluation problem

- problem ( 평가 문제 ) : HMM($\lambda$)과 O가 주어졌을 때 Observable sequence O'의 확률 (이는 모델이 얼마나 잘 관측 데이터를 설명하는지 평가하는 데 중요)
  - Solution : `Forward Algorithm`
  - example : 오늘 산책, 내일 산책, 모레 연구, 글피 쇼핑할 확률은?
  - Forward probability (전방 확률, $\alpha_{t}(i)$) = $p(O| \lambda) = \sum_{i=1}^{n} \alpha_{T}(j)$
    - 순차적으로 (뒤 -> 앞으로) 계산
    - 1. $a_1(i) = \pi_i b_i(o_1)$ , $1 \leq i \leq n$
    - 2. $a_{t}(i) = [\sum_{j=1}^{n} a_{t-1}(j) a_{ji}] b_i(o_t)$, $2 \leq t \leq T, 1 \leq i \leq n$
      - $바로 직전 상태 state * event가 관측될 확률$
- Forward probability는 주어진 Sequence O가 HMM에 속할 확률 문제에 활용 가능
  - HMM1과 HMM2가 있을 때 어느 HMM에 속할 확률이 높을지?
  - Sequence classification 문제에 활용 가능

### Decoding problem

- Problem : HMM($\lambda^*$)와 O가 주어졌을 때 최적의 S결정(가장 그럴싸한 은닉 상태 시퀀스 결정)

  - Solution : `Viterbi Algorithm`
  - example : 오늘 산책, 내일 산책, 모레 연구, 글피 쇼핑을 했다면 각 날들 날씨는?
  - Viterbi Algorithm for Decoding problem
    - 전방 계산 (매 단계에서 최적 경로 기록)
      - $v_{t}(i) = max_{q_1, q_2, ..., q_{t-1}} p(o_1, o_2, ..., o_t, q_1, q_2, ..., q_{t-1}, q_t = s_i | \lambda)$ = $max_{1 \leq j \leq n} [v_{t-1}(j) a_{ji}] b_i(o_t)$ / $2 \leq t \leq T, 1 \leq i \leq n$
      - $v_1(i) = \pi_i b_i(o_1)$
    - 최적 경로 추적
      - $ \hat q*{T} = argmax*{1 \leq j \leq n} (v\_{T}(j))$
      - $ \hat Q*{t} = (\hat q*{1}, \hat q*{2}, ..., \hat q*{t})$

- Learning problem

#### Evaluation vs Decoding

- Forward Algorithm for Evaluation : 가능한 모든 경우의 확률 합
  - Observable state의 확률을 구하는 것이 목표
- Viterbi Algorithm for Evaluation : 가능한 모든 경우의 확률 **최대** 가장 그럴싸한 상태를 찾는 것
