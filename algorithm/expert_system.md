# 🚀 학습 키워드

- 지식의 개념 이해
- 규칙기반 전문가 시스템의 구조와 요소
- 전문가 시스템에서 추론이 이루어지는 과정 이해
- 전문가 시스템의 장단점 이해

#### 전문가 시스템 개발의 배경

- 인공지능 초창기 : 이 세상에 존재하는 모든 문제 -> "탐색"으로 해결할 수 있는 시스템을 만들고자 노력
- 실제적인 문제를 해결하기 위해 인공지능 연구자들은 보다 제한된 문제에 역량을 집중하는 것이 필요하다는 것을 깨달았다 -> 전문가 시스템 (expert system)
- `지식이 추론 기법 만큼 중요하다는 것을 깨닫게 되었다`

---

# 📝 정의, 개념

## 전문가 시스템의 의의

- 전문가 시스템 : 기존의 절차적 코드가 아니라, `규칙으로 표현되는 지식을 통해 추론`함으로써 복잡한 문제를 해결하도록 설계
- 전문가 시스템은 인공지능(AI) 소프트웨어의 최초의 성공적인 형태

## 전문가 시스템의 구성 요소

- 지식 베이스, 추론 기관, 사용자 인터페이스
- 데이터, 정보, 지식 : 데이터는 단순한 사실, 정보는 데이터를 가공한 결과, 지식은 정보를 통해 얻은 경험과 규칙

### 지식 표현

- 지식 : 경험이나 교육을 통해 얻어진 전문적인 이해와 체계화된 문제 해결 능력
- 어떤 주제나 분야에 대한 이론적 또는 실제적인 이해
  - 암묵지 : 개인의 경험, 지식, 노하우, 직관 등을 통해 쌓은 지식
  - 형식지 : 비교적 쉽게 형식을 갖추어 표현될 수 있는 지식
  - 절차적 지식(Procedural Knowledge) : 문제 해결의 절차 기술
  - 선언적 지식(Declarative Knowledge) : 어떤 대상의 성질, 특성이나 관계 서술
- 지식 표현 방법 : 컴퓨터를 통한 지식 표현 및 처리
  - 프로그램이 쉽게 처리할 수 있도록 정형화된 형태로 표현
    - 생성 규칙 또는 규칙 : 절차적
    - 술어 논리 : 선언적
    - 의미망 : 선언적
    - 프레임 : 선언적

### 생성 규칙

- 생성 규칙 : `IF-THEN` 형태로 표현되는 규칙 (~이면 ~이다, ~하면 ~하다)
- 규칙 획득 및 표현 : 대상, 속성, 행동 또는 판단의 정보 추출
- 표현 : `IF (조건) THEN (결과)` 형태로 표현
  - IF 부분 : 주어진 정보나 사실에 대응될 조선/전제/상황
  - Then 부분 : 조건이 만족될 때의 판단이나 행동
- 종류 : 인과관계, 추천, 지시(명령), 전략, 휴리스틱

## 규칙 기반 시스템 (rule-based system)

- 지식을 규칙의 형태로 표현
- 주어진 문제 상황에 적용될 수 있는 규칙들을 사용하여 문제에 대한 해를 찾도록 **지식 기반 시스템**
- **전문가 시스템**을 구현하는 전형적인 형태 : 특정 문제 영역에 대해서 전문가 수준의 해를 찾아주는 시스템

### 추론

- 구축된 지식과 주어진 데이터나 정보를 이용하여 새로운 사실을 생성하는 것
- 추론 방법 : 순방향 추론, 역방향 추론
  - 순방향 추론 (forward chaining) : 알려진 사실로부터 출발하여 결론을 이끌어 내는 방법
  - 역방향 추론 (backward chaining) : 목표를 설정하고 추론 엔진은 이를 증명하는 증거를 찾는 방법
- 충돌해법 : 동일한 사실이 입력되어도 서로 상반된 결론을 내리는 규칙이 저장되어 있다면 어떻게 해야할까?
  - **우선순위 부여** : 각 규칙에 우선 순위를 부여하고 가장 높은 우선순위를 가진 규칙을 점화하는 방법

### 전문가 시스템(규칙)의 장단점

장점

- if-then을 사용하는 규칙은 인간 전문가의 지식을 표현하는 자연스러운 방법
- 전문가 시스템에서는 지식베이스와 추론엔진이 분리, 따라서 다른 영역에도 쉽게 적용 가능

단점

- 지식을 학습할 수 없다.(새로운 사실 도축가능 하지만 새로운 지식 학습 못함)
- 탐색이 비효율적 -> 탐색 알고리즘으로 극복
- 규칙이 많아지게 되면 유지보수하는 것이 어려워짐

---

# ✨ REMIND (키워드)

- 전문가 시스템 : 전문지식을 가진 시스템
- 전문적인 지식표현 : 규칙/ 술어논리/ 의미망/ 프레임 등의 방법
- 규칙사용 지식표현 : 대상/속성/판단을 정의하여 기술
- 복잡한 질의에 대답하거나 새로운 사실 추론 : 순방향/역방향 추론 사용
- 규칙의 충돌 : 충돌해법 사용