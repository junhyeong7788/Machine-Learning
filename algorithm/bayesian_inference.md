# 🚀 학습 키워드

## 정의

- 확률은 빈도론적 확률과 베이지안으로 분류

  - 빈도론 : 반복적인 사건의 빈도
  - 베이지안 : 확률을 '주장에 대한 신뢰도'

- 확률 정의 : $\frac{사건 시행 횟수}{관심 있는 사건 발생 횟수}$
- 베이지안 확률 (Bayesian Probability) : 믿음의 정도로 관심 사건의 발생 확률의 **신뢰도**를 추론
- 베이즈 추론 : 모든 가설에 따라 사후 확률을 계산한 후 확률이 가장 높은 쪽을 선택하여 결정하는 방법

---

# 📝새로 배운 개념

## 조건부 확률 (Conditional Probability)

- 어떤 사건 A가 발생한 상황에서, 다른 사건 B가 발생할 확률
- $P(B|A) = \frac{P(A \cap B)}{P(A)}$
  - $P(B|A)$ : 사건 A가 발생했을 때 사건 B가 발생할 확률
  - $P(A \cap B)$ : 사건 A와 사건 B가 동시에 발생할 확률 (교집합)
  - $P(A)$ : 사건 A가 발생할 확률

## 베이즈 정리 (Bayes' Theorem)

- 조건부 확률에서 베이즈 정리 수식 유도

  - 1. $P(A \cap B) = P(B|A)P(A)$
  - 2. $P(A \cap B) = P(B \cap A)$
  - 3. $P(B \cap A) = P(A|B)P(B)$
  - $P(B|A) = \frac{P(A|B)P(B)}{P(A)}$
  - 베이지안 확률 또는 베이즈 정리라고 부른다. 결국 조건부 확률을 써둔 것이다.

- 베이즈 정리
  - $P(H|E) = \frac{P(E|H)P(H)}{P(E)}$
    - $P(H|E)$ : 사후 확률 (Posterior Probability) : E가 발생했다는 조건하에 H라는 결과가 나올 확률
    - $P(E|H)$ : 가능도 (Likelihood) : E가 발생했다는 조건 하에 H라는 결과가 나올 확률
    - $P(H)$ : 사전 확률 (Prior Probability) : 어떤 사건이 발생했다는 주장에 관한 신뢰도
    - $P(E)$ : 증거 (Evidence) : 새로운 정보(=관측한 사실)
    - H (hypothesis) : 가설 ( = 어떤 사건이 발생했다는 주장)
    - E (evidence) : 새로운 정보 ( = 관측한 사실 )

## 베이즈 추론

- 모든 가설에 따라 사후 확률을 계산한 후 확률이 가장 높은 쪽을 선택하여 결정하는 방법
- 베이지안 추론에 나타나는 세개의 확률을 정리
  - $P(H_{i}|D) \alpha P(D|H_{i})P(H_{i})$
- 결국 베이지안 추론은 Posterior확률이 최대인 값을 찾는 과정
- 베이지안 추론은 MAP (Maximum A Posteriori) 문제라고도 한다.
  - $argmax_{h}(P(H_{i}|D))$
  - 외부로 드러난 관찰 (증상)에 기반해서 숨겨진 가설을 추론할 때 사용한다.
  - 즉, 관찰된 현상을 통해 그 속에 숨겨진 본질을 찾는 것이 목표

---

# ✨ REMIND

- 조건부 확률 (Conditional Probability)
- 베이즈 정리 (Bayes' Theorem)
- 베이즈 추론 (Bayesian Inference)
