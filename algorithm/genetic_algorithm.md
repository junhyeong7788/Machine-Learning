# 🚀 학습 키워드

- 유전자 알고리즘의 기본 개념 이해
- 유전자 알고리즘의 여러 문제
- 유전자 프로그래밍의 작동방식

## 정의

- 유전자 알고리즘 : 생물체의 염색체가 유전되는 현상에서 영감을 얻은 최적화 알고리즘으로서 적자생존 원칙에 기반을 두고 교차, 돌연변이, 도태 등의 과정을 통하여 우성 유전자만이 살아남는 자연계의 현상을 알고리즘으로 만든 것

## 키워드 (GA terminology)

- Chromosome : 유전자 알고리즘에서 사용되는 개체의 표현 방식
- gene : 개체의 특성을 나타내는 유전자
- Population : 유전자 알고리즘에서 사용되는 개체의 집합
  - individuals : 개체
- genetic operators : 유전자 알고리즘에서 사용되는 교차, 돌연변이, 도태 등의 과정
  - selection : 개체를 선택하는 과정
  - crossover : 두 개체의 유전자를 교환하는 과정
  - insertion : 개체를 삽입하는 과정
  - mutation : 개체의 유전자를 변이시키는 과정

---

# 📝새로 배운 개념

## Heuristic Search Techniques 한계

1. Local Minima 문제

- 최적의 해를 구하지 못했지만 최적이라고 판단하여 탐색을 종료

2. 시스템 최적화 문제

- 하나의 시스템은 연속적인 단일 변수만 존재하는 것이 아니라 discrete, categorical, integer, Boolean 등 다양한 변수가 복잡적으로 있다.
- 다변수 시스템에서 모든 변수의 최적의 해를 찾는 것은 어려운 일

### Chromosome

- 각 염색체는 하나의 솔루션을 표현
- 보통의 경우 염색체의 각 bit는 0 또는 1로 표현하며, 유전자 하나를 의미
- 주어진 염색체의 유전자를 실제 어떤 값(대체 유전자)로 표현하는 것을 Encoding이라 함
- 따라서 염색체는 `Encoding된 정보`(이진수) 이기 때문에 `Decoding이 필요`(10진수)

## Canonical Genetic Algorithm

```
           START
             |
    1. Initialization
             |
------------>|
|            |
|   2. Fitness evaluation
|            |
|      3. Selection
|            |
|      4. Crossover
|            |
|      5. Mutation
|            |
-----------STOP?
             |
            END
```

### Fitness evaluation

- 각 개체 $x$는 성능 측정치로서의 적합도 값 $f(x)$를 할당
- TSP(외판원 문제)에서 적합도는 보통 여행 비용 (시간, 거리, 가격 등)을 의미
  - 비용이 낮을수록 더 좋은 경로로 간주
- 올바른 적합도 함수를 선택하는 것은 매우 중요하지만, 동시에 꽤 어렵다

### Selection

- 현재 개체군에서 개체를 선택하여 재생산을 위한 `교배 풀(mating pool)`을 구성
- Gaol : `교차(crossover)`를 위한 부모를 선택하는 것

1. Ranking : 개체들을 적합도 순으로 정렬한 뒤, 순위에 따라 선택 확률 할당
   - 높은 순위를 가진 개체가 선택될 확률이 더 크다
   - $D = \sum_{j \in P} \frac{1}{j}$
   - $P_{k} = \frac{1}{k} * D^{-1}$
     - $k$ : 순위
     - $D$ : 정규화 계수
2. Proportional to Fitness value (적합도 기반 선택)
   - 개체의 선택확률은 적합도 점수에 비례
   - $\bar{F} = \sum_{j \in P} Fitness(j)$
   - k번째로 적합한 개체를 선택하여 부모로 삼을 확률 : $P_{k} = Fitness(k) * \bar{F}^{-1}$
   - 각 개체의 적합도를 계산하여, 전체 적합도 합계 대비 비율로 선택확률을 결정
3. Roulette Wheel Selection
   - 각 개체의 적합도 비율에 따라 룰렛휠에 "조각"을 할당 -> `각 개체의 선택확률을 누적하여 계산한 것`
     - 컴퓨터에는 룰렛판이 없기때문에 [0, 1.0]사이의 값을 기반으로 선택
   - 무작위 숫자를 생성하여 해당 숫자에 해당하는 개체를 선택
   - 적합도가 높을수록 더 큰 조각을 차지하여 선택될 확률이 증가
4. Tournament Selection
   - 무작위로 선택된 n개의 개체 중 적합도가 가장 높은 개체를 선택
   - 각 토너먼트를 반복하여 새로운 생존자 집단을 구성
   - 계산이 간단, 적합도에 대한 선택 입력을 조정

### Crossover

- 교차란? : 두 개의 부모 염색체를 선택하여 유전자를 교환함으로써 새로운 자식 염색체를 생성하는 과정
  - 부모 선택 : 교배 풀(mating pool)에서 두 부모를 선택
    - 교배 풀 : 다음 세대의 개체를 생성하기 위해 부모 개체로 선택된 개체들의 집합
  - 확률적 실행 : 교차 연산은 교차 확률 $P_{C}$에 따라 수행
    - $P_{C} = 1$ : 교차가 항상 수행
    - $P_{C} < 1$ : 교차가 수행되거나 수행되지 않음
    - $P_{C} = 0$ : 교차가 수행되지 않음
  - 교차점(Crossover point)은 무작위로 선택, 선택된 교차점을 기준으로 두 부모 간의 문자열이 교환됨

1. Single Point Crossover
   - 하나의 교차점을 선택하여 부모의 유전자를 교환
2. Multi Point Crossover
   - 두 개 이상의 교차점을 선택하여 교환
   - 더 복잡한 유전적 조합 생성 가능

- 그 외 : Path relinking(경로 재연결 기법), permutation(순열 기반 교차), random keys approach(무작위 키를 기반 교차) 등...

### Mutation

- 변이 연산자 (`변이확률`) : 유전자 단위로 적용되며, 각 유전자는 확률 $P_{m}$에 따라 변이를 겪음
- 변이 연산이 유전자에 발생하면, 해당 유전자의 값이 반전(flipped) 됨 / `0 -> 1, 1 -> 0`
- 다른 대체 방식
  - 기존 개체 모두 대체
  - 변이를 통해 생성된 자식이 부모를 대체
  - 무작위로 기존 개체를 대체
  - 기존 개체 중 변이된 개체와 가장 유사한 것을 대체
- mutation을 안하면 local minima에 빠질 수 있다. mutation이 존재함으로 전역 최적점을 찾기가 쉽다.

### GA Stopping Criteria (종료 조건)

- x세대 수가 완료됨 : 가장 일반적인 종료 조건
- 개체 성능의 평균 편차 (mean deviation) : 집단 내 개체들의 성능 편차가 임계값 $\sigma_{J} < x$ 아래로 떨어질 때 (유전적 다양성이 줄어들었음을 나타냄)
- 정체 (Stagnation) : 한 세대에서 다음 세대로의 개선이 없거나 매우 미미한 경우, $J_{n+1}-J_{n} < X$ (최적화가 더 이상 진행되지 않음)

---

# ✨

### GA와 전통적 방법의 차이점

- GA는 매개변수 값의 인코딩에 대해 작동, 실제 매개변수 값 자체에 대해 작동하지 않음
- GA는 단일 해답이 아닌 해답의 집단을 기반으로 작동
- GA는 목표 함수를 기반으로 한 적합도 값만 사용
- GA는 확률적 계산을 사용하며, 전통적 방법은 결정론적 계산을 사용
- GA는 이산적이거나 혼합된 탐색 공간을 처리하는 데 효율적

### GA가 유용한 경우

- 최적화 문제에 적용 가능
  - 단 결정론적 문제에서는 GA사용은 비효율적
- GA는 확률적 특정으로 인해 계산 비용 예상됨
- 전통적 방법으로 해결할 수 없는 제약 조건 만족 문제와 같이 완벽하게 최적의 해답이 필요하지 않은 문제에서 자주 사용됨

## Summary

- 자연계의 진화 현상을 인공지능에 응용한 것이 유전자 프로그래밍
- 유전자 알고리즘에서는 개체 집단을 만들어 적합도를 평가하고 선택이나 교차, 돌연변이 연산자를 적용하여 새로운 해집단을 생성, 이 과정을 일정한 횟수만큼 되풀이함