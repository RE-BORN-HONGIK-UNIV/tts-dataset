# 음성 비유창성 탐지 모델: 데이터·라벨링·학습·평가 설계서

> 프로젝트: 고립청년 대상 대면 면접훈련 서비스
>
> 문서 목적: 1단계 음성 분석 모델의 데이터 구성, 라벨 체계, 실제·합성 데이터 활용, 학습 및 평가 원칙을 팀의 공통 기준으로 남긴다. 이 문서는 구현 기준이자 향후 학술제 발표 준비를 위한 방법론 학습 노트이다.

---

## 1. 프로젝트 맥락과 목표

본 프로젝트는 고립청년이 대면 면접을 연습할 때, 음성에서 나타나는 비유창성 및 불안 관련 음향 신호를 분석해 훈련 피드백을 제공하는 서비스 개발을 목표로 한다.

### 1.1 단계별 구성

| 단계 | 분석 대상 | 주요 방법 |
|---|---|---|
| 1단계 | 면접 음성 | 채움말·멈춤은 규칙 기반 분석, 연장·떨림·에너지변동은 CNN 기반 분류 |
| 2단계 | 시선·표정 | 시선 및 표정 기반 불안도 측정 |

이 문서는 **1단계 음성 분석 중 연장(prolongation), 떨림(tremor), 에너지변동(energy variation)** 데이터와 모델 학습 방향을 다룬다.

### 1.2 서비스상 해석의 범위

모델은 임상적 진단 또는 정신건강 상태의 판정을 목적으로 하지 않는다. 면접 연습 상황에서 나타나는 말하기 특성을 관찰하여 사용자가 스스로 훈련할 수 있도록 돕는 보조 피드백 신호를 제공한다.

따라서 서비스 문구는 다음처럼 표현한다.

```text
권장: "발화 후반부에서 음량이 낮아지는 경향이 관찰되었습니다."
비권장: "불안장애가 있습니다." / "불안도가 높습니다."
```

---

## 2. 과업과 라벨 구조

### 2.1 상위 라벨

실제 일반인 음성 데이터에는 현재 다음의 상위 라벨이 존재한다.

```text
normal
prolongation
tremor
energy_variation
```

| 라벨 | 의미 | 모델 처리 방향 |
|---|---|---|
| `normal` | 목표 현상이 뚜렷하지 않은 일반 발화 | 각 탐지 과업의 기준 클래스 |
| `prolongation` | 특정 음소·음절이 비정상적으로 길게 이어지는 발화 | 연장 탐지 모델의 양성 클래스 |
| `tremor` | 음성의 진폭·주파수 등에 반복적 흔들림이 나타나는 발화 | 떨림 탐지 모델의 양성 클래스 |
| `energy_variation` | 발화 구간에서 에너지 변화가 관찰되는 발화 | 에너지 모델에서 세부 방향 라벨로 확장 |

### 2.2 에너지변동 하위 라벨

에너지변동은 단순히 “변화가 있다/없다”를 넘어서, 변화 방향을 구분한다.

```text
normal_energy
energy_fade_in
energy_fade_out
energy_ambiguous
recording_issue
```

| 하위 라벨 | 작업 정의 | 학습·평가 사용 |
|---|---|---|
| `normal_energy` | 분석 구간에서 뚜렷한 단조 에너지 상승·하강이 관찰되지 않음 | 사용 |
| `energy_fade_in` | 발화 전반부에서 후반부로 갈수록 에너지가 전반적으로 증가하는 경향 | 사용 |
| `energy_fade_out` | 발화 전반부에서 후반부로 갈수록 에너지가 전반적으로 감소하는 경향 | 사용 |
| `energy_ambiguous` | 변화는 있으나 단조 상승·하강으로 신뢰성 있게 분류하기 어려움 | 초기 3클래스 학습·평가에서 제외 |
| `recording_issue` | 무음, clipping, 잡음, 마이크 거리 변화 등 녹음 품질 문제로 판정 불가 | 학습·평가에서 제외 |

라벨의 계층 관계는 다음과 같다.

```text
energy_fade_in  ┐
                ├── energy_variation
energy_fade_out ┘

energy_ambiguous ── energy_variation에 포함될 수 있으나,
                    초기 3클래스 학습에서는 별도 보류한다.
```

> **주의**: `prolongation`과 `tremor`는 에너지모델의 normal 또는 energy 라벨에 임의로 포함하지 않는다. 한 발화에 여러 현상이 공존할 수 있으므로, 초기에는 과업별로 데이터를 분리해 관리한다.

---

## 3. 데이터 출처와 역할

### 3.1 실제 일반인 음성

실제 일반인 음성은 서비스가 최종적으로 동작해야 하는 목표 도메인이다. 현재는 화자(`speaker_id`)가 구분되어 있고, `normal`, `prolongation`, `tremor`, `energy_variation` 상위 라벨을 보유한다.

실제 데이터는 다음에 사용한다.

- 실제 환경의 음색, 발화 습관, 녹음 조건, 배경 소음 반영
- 실제 에너지변동의 fade-in/fade-out 세분화 대상
- 학습 데이터의 일부
- 모델 선택을 위한 validation 데이터
- 최종 일반화 성능 확인을 위한 test 데이터

### 3.2 합성 음성

합성 음성은 부족한 데이터를 보완하고, 특히 명확한 방향성을 가진 에너지 변화 사례를 만들기 위해 사용한다.

합성 에너지 라벨은 아래 세 가지를 기본으로 한다.

```text
normal_energy
energy_fade_in
energy_fade_out
```

합성 데이터는 다음 역할을 맡는다.

- 방향이 명확한 에너지 패턴을 충분히 제공
- 실제 데이터가 적을 때 CNN의 초기 특징 학습 보조
- 클래스 불균형 완화
- 실제 데이터와 결합한 공동학습 데이터

합성 데이터는 실제 데이터의 대체물이 아니다. 실제와 합성 음성 사이에는 화자 특성, 녹음 조건, TTS 특성, 발화 자연스러움 등에서 분포 차이(domain gap)가 존재할 수 있다. 따라서 최종 성능은 항상 실제 데이터로 판단한다.

### 3.3 데이터 역할 요약

| 구분 | 라벨 수준 | 핵심 역할 | 최종 성능 평가에 사용 여부 |
|---|---|---|---|
| 실제 데이터 | 상위 라벨 + 검증 후 세부 라벨 | 학습, 검증, 실제 성능 평가 | 사용 |
| 합성 데이터 | 세부 라벨이 명확함 | 사전학습, 증강, 공동학습 | 미사용 |

---

## 4. 실제 에너지변동의 세분화

### 4.1 왜 세분화가 필요한가

현재 실제 데이터의 `energy_variation`은 에너지 변화의 존재 여부만 나타내며, fade-in과 fade-out의 방향은 구분하지 않는다. 반면 합성 데이터는 `normal / fade_in / fade_out`의 세부 라벨을 가진다.

실제 데이터도 fade 방향으로 세분화하면 다음 장점이 있다.

- 실제·합성 데이터의 라벨 공간을 맞출 수 있다.
- 실제 음성에서 fade-in과 fade-out을 각각 검증할 수 있다.
- 서비스에서 방향성 있는 피드백 후보를 제시할 수 있다.
- 합성 데이터에만 의존해 세부 클래스를 학습하는 위험을 줄일 수 있다.

### 4.2 반자동 라벨링 원칙

자동 규칙만으로 실제 음성의 fade 방향을 확정하지 않는다. 자동 규칙은 사람 검토 대상을 빠르게 찾는 **후보 생성기(candidate generator)** 로 쓴다.

```text
기존 실제 energy_variation 구간
→ 음성 품질 및 VAD 확인
→ 에너지 궤적 자동 분석
→ fade_in / fade_out / ambiguous 후보 부여
→ 복수 청취자의 블라인드 청취 판단
→ 고신뢰 최종 라벨 확정
```

음성 운율 분석에서는 intensity뿐 아니라 pitch(F0), 지속시간, silence 등 여러 음향 특징이 관련될 수 있으며, 자동 prosody labeling 연구들도 자동 분석 결과를 수동 주석과 비교하여 검증하는 절차를 둔다. 자동 라벨은 수작업 주석을 대체하는 절대 정답이 아니라 라벨링 효율을 높이는 보조 수단으로 취급한다.

### 4.3 자동 후보 생성 특징

`energy_variation`으로 라벨된 구간을 대상으로 아래 특징을 추출한다.

```text
1. VAD로 확인한 유성 발화 구간
2. 짧은 프레임 단위 RMS 에너지 또는 log-energy
3. 평활화된 에너지 궤적
4. 전반부와 후반부 평균 에너지 차이
5. 시간 대비 에너지 선형회귀 기울기
6. 중간 구간의 급격한 반전·비단조성 여부
7. clipping, 매우 짧은 구간, 낮은 SNR 등 품질 문제
```

개념적으로는 아래와 같이 계산할 수 있다.

\[
\Delta E_{dB} = \overline{E}_{\text{후반부}} - \overline{E}_{\text{전반부}}
\]

\[
E(t) = \beta_0 + \beta_1 t + \epsilon
\]

- \(\Delta E_{dB} > 0\) 및 \(\beta_1 > 0\): `fade_in` 후보
- \(\Delta E_{dB} < 0\) 및 \(\beta_1 < 0\): `fade_out` 후보
- 방향성이 약하거나, 기울기와 전·후반 차이의 방향이 일치하지 않거나, 중간 반전이 큰 경우: `ambiguous` 후보

> **판정 임계값 주의**: 위 식은 방향을 계산하기 위한 작업 정의이다. 현재 특정 dB 차이 또는 기울기 값을 실제 자연 음성의 보편적 임계값으로 확정하지 않는다. 합성 음성 생성에 사용한 변화량은 합성 강도 제어값이며 실제 라벨 판정 근거가 아니다. 실제 라벨링 파일럿에서 사람 청취 결과와 특징 분포를 비교한 뒤, 프로젝트 내부의 잠정 threshold를 정한다.

### 4.4 사람 검증 절차

자동 후보 생성 후 최소 2명, 가능하면 3명의 청취자가 독립적으로 라벨링한다.

청취자가 선택할 라벨은 다음과 같다.

```text
fade_in
fade_out
ambiguous
recording_issue
```

검증 원칙:

- 자동 알고리즘의 후보 라벨은 청취자에게 보여 주지 않는다.
- 청취자는 같은 구간을 독립적으로 판단한다.
- 가능하면 원본 파형·스펙트로그램 없이 음성 중심으로 우선 판단하고, 품질 이슈 확인이 필요할 때만 보조 시각화를 사용한다.
- 의견 불일치 사례는 억지로 다수결 정답을 만들기보다 `ambiguous`로 보류할 수 있다.
- 고신뢰 라벨만 초기 방향 분류 모델의 학습·평가 데이터로 사용한다.

### 4.5 합의 기록

라벨링 CSV에는 최소한 다음 필드를 남긴다.

```csv
sample_id,speaker_id,recording_id,start_sec,end_sec,original_label,auto_candidate,rater_1,rater_2,rater_3,final_label,agreement,quality_note
```

예시:

```csv
spk023_utt07,spk023,rec023,12.40,15.80,energy_variation,fade_out,fade_out,fade_out,,fade_out,2/2,
spk031_utt02,spk031,rec031,5.10,8.20,energy_variation,fade_in,fade_in,ambiguous,,ambiguous,disagree,mid-utterance emphasis
```

라벨러 간 일치도는 Cohen's kappa(2명) 또는 Fleiss' kappa(3명 이상)를 함께 기록하는 것을 권장한다. 다만 이 수치는 실제 라벨링 파일럿 후 산출하며, 현 시점에 목표 수치를 임의로 확정하지 않는다.

---

## 5. 데이터 분할 원칙

### 5.1 화자 단위 분할

실제 데이터는 파일 단위가 아니라 반드시 **화자 단위**로 train, validation, test를 분리한다.

```text
Real train        : 학습용 실제 화자
Real validation   : 모델 선택용 실제 화자
Real test         : 최종 평가용 실제 화자
Synthetic train   : 합성 데이터
```

같은 화자의 음성이 train과 validation/test에 동시에 있으면, 모델이 현상 자체가 아니라 음색·말투·마이크 환경을 외울 수 있다. 그러면 실제 성능보다 높은 평가값이 나오는 data leakage가 발생한다.

### 5.2 예시 분할

실제 화자가 약 100명인 경우의 예시다.

| 분할 | 화자 수 예시 | 용도 |
|---|---:|---|
| Real train | 70명 | 모델 파라미터 학습 |
| Real validation | 15명 | epoch, threshold, 모델 구조, sampling 방식 선택 |
| Real test | 15명 | 최종 성능 보고 |

화자 수가 적거나 각 클래스가 극도로 불균형할 때는 위 비율을 기계적으로 적용하지 않는다. 각 분할에 `normal`, `prolongation`, `tremor`, `energy_variation` 및 에너지 세부 라벨이 가능한 한 포함되도록 **화자 단위 층화 분할**을 적용한다.

### 5.3 합성 데이터의 위치

합성 데이터는 **train에만** 넣는다.

```text
허용: synthetic train + real train
금지: synthetic validation, synthetic test, real test에 합성 혼입
```

validation과 test를 실제 음성으로 구성해야, 서비스가 실제 사용자 음성에서 얼마나 작동할지를 평가할 수 있다.

---

## 6. 학습 방향

### 6.1 기본 방침

현재는 데이터가 부족하므로 실제와 합성 데이터를 모두 활용한다.

```text
Training data
= 고신뢰 실제 train 데이터
+ 합성 normal / fade_in / fade_out 데이터
```

이는 “합성과 실제의 비율 최적화를 전혀 고려하지 않는다”는 뜻이 아니다. 초기 연구 단계에서는 먼저 전량을 활용한 기준 모델(baseline)을 만들고, 결과가 나온 뒤 최소한의 비교 실험으로 sampling 전략의 영향을 확인한다는 뜻이다.

### 6.2 권장 학습 단계

```text
1단계. 합성 데이터 사전학습
      - normal_energy / energy_fade_in / energy_fade_out
      - 방향성 있는 에너지 패턴의 초기 특징 학습

2단계. 합성 + 실제 train 데이터 공동학습
      - 실제 음성 특성에 적응
      - 실제 validation 기준으로 early stopping

3단계. 선택적 실제 데이터 fine-tuning
      - 실제 train 데이터만 사용
      - 낮은 learning rate 적용
      - real validation 성능이 좋아질 때만 최종 채택

4단계. 실제 test 단 1회 평가
      - 최종 모델과 threshold가 고정된 뒤에만 실행
```

위 단계는 권장 초기안이다. 최종 선택은 validation 결과를 바탕으로 하며, test 결과를 보고 반복적으로 구조나 hyperparameter를 바꾸지 않는다.

### 6.3 에너지 모델의 출력

초기 에너지 방향 모델은 다음 3클래스를 출력한다.

```text
normal_energy
energy_fade_in
energy_fade_out
```

실제 데이터의 fade 방향 세분화가 아직 충분하지 않은 경우에는, 모델의 두 방향 확률을 합쳐 상위 이진 과업으로도 평가할 수 있다.

\[
P(\text{energy_variation}) = P(\text{energy_fade_in}) + P(\text{energy_fade_out})
\]

이때 실제 `normal`과 실제 `energy_variation`을 비교하여 다음을 평가한다.

```text
normal vs energy_variation
```

방향별 성능(`fade_in` 대 `fade_out`)은 실제 방향 라벨이 사람 검증으로 확보된 후에만 보고한다.

---

## 7. 실제·합성 데이터 비율

### 7.1 교수님 피드백의 해석

“데이터가 너무 부족하니 구분하지 말고 일단 다 학습시키라”는 피드백은 현재 단계에서 타당하다. 실제·합성 데이터를 모두 보존하고 학습에 참여시키는 것을 기본 원칙으로 한다.

다만 실제 데이터가 수백 개이고 합성 데이터가 수만 개처럼 압도적으로 많으면, 파일을 무작위로 섞었을 때 모델이 대부분의 업데이트에서 합성 데이터만 보게 된다. 이 경우 실제 음성이 학습에서 묻힐 수 있다.

### 7.2 초기 운영값: source-balanced mini-batch

합성이 실제보다 훨씬 많을 때는 모든 데이터를 보존하면서, 학습 미니배치에서 출처를 균형화한다.

```text
잠정 기본값
- 실제 음성: 50%
- 합성 음성: 50%
```

이 1:1은 학술적으로 보편 타당성이 확립된 최적값이 아니다. 본 프로젝트에서의 **잠정 운영값**이며, 실제 validation 성능을 통해 유지 또는 변경한다.

또한 출처(source) 균형과 클래스(class) 균형은 별개다.

```text
source balance: 실제 / 합성 비율
class balance : normal / fade_in / fade_out 비율
```

예를 들어 실제 데이터에서 `normal`이 지나치게 많으면, 실제가 batch의 50%여도 normal만 주로 학습될 수 있다. 따라서 class-balanced sampler 또는 class weight를 적용할지 validation 결과로 판단한다.

---

## 8. 최소 실험 설계

초기에는 광범위한 비율 탐색 대신 아래 네 개 실험을 수행한다.

| ID | 학습 데이터 | 목적 |
|---|---|---|
| E1 | 합성만 | 합성에서 학습한 패턴이 실제 음성으로 전이되는 수준 확인 |
| E2 | 실제만 | 소량 실제 데이터만으로 학습한 기준선 확인 |
| E3 | 실제 + 합성 전량 혼합 | 기본 모델 및 전량 활용 원칙 검증 |
| E4 | 실제 + 합성 + source-balanced batch | 합성 과대표집이 실제 성능에 미치는 영향 확인 |

### 8.1 실험 선택 규칙

- 모델 선택은 `Real validation` 성능으로 한다.
- `Real test`는 최종 모델이 정해진 뒤 단 1회 사용한다.
- 각 실험은 동일한 실제 화자 분할, 동일한 seed 정책, 동일한 평가 코드로 비교한다.
- 데이터 수가 매우 적어 결과 분산이 큰 경우, 여러 random seed 결과의 평균과 표준편차를 기록한다.

### 8.2 해석 예시

| 관찰 결과 | 해석 및 다음 행동 |
|---|---|
| E3가 E1·E2보다 좋음 | 실제와 합성을 결합한 데이터 보완 효과가 확인됨 |
| E4가 E3보다 좋음 | 합성 비중이 과도했을 가능성. source-balanced batch 채택 고려 |
| E1은 낮고 E2·E3은 높음 | 합성-실제 domain gap 존재. 실제 데이터 공동학습이 중요함 |
| E2와 E3가 유사함 | 합성 데이터 생성 품질, 라벨 정의, sampling을 재점검 |

---

## 9. 평가 원칙

### 9.1 최종 성능의 기준

최종 서비스 성능은 **실제 test 화자 데이터**로만 보고한다.

- 합성 test 성능은 합성 규칙을 재현하는 정도만 보여 줄 수 있다.
- 실제 test 성능이 서비스 적용 가능성을 판단하는 핵심 지표다.

### 9.2 권장 지표

에너지변동 이진 평가와 방향 분류 평가에서 다음 지표를 기록한다.

| 지표 | 의미 |
|---|---|
| Precision | 모델이 양성이라고 판단한 사례 중 실제 양성 비율 |
| Recall | 실제 양성 사례 중 모델이 찾은 비율 |
| F1-score | Precision과 Recall의 조화평균 |
| PR-AUC | 양성 클래스가 적을 때 threshold 전반의 성능을 보는 지표 |
| ROC-AUC | 분류 점수의 전반적 분리 능력 |
| Confusion matrix | 어떤 클래스가 어떤 클래스로 혼동되는지 확인 |

실제 방향 라벨이 없을 때는 `normal` 대 `energy_variation`의 이진 성능만 보고한다. `fade_in`과 `fade_out`의 방향별 정확도는 실제 방향 정답이 확보된 뒤에만 보고한다.

### 9.3 Threshold 선택

분류 threshold는 test 성능을 가장 높이는 값으로 정하지 않는다. validation 세트에서 정한 threshold를 고정한 뒤 test에 적용한다.

```text
올바른 순서:
train → validation에서 threshold 선택 → threshold 고정 → test

잘못된 순서:
train → test에서 threshold 탐색 → 가장 좋은 test 수치 보고
```

---

## 10. 데이터셋과 실험 메타데이터

### 10.1 권장 디렉터리 구조

```text
project-root/
├── data/
│   ├── raw/
│   │   ├── real/
│   │   └── synthetic/
│   ├── processed/
│   ├── metadata/
│   │   ├── real_manifest.csv
│   │   ├── synthetic_manifest.csv
│   │   ├── energy_annotation.csv
│   │   └── splits/
│   │       ├── real_train_speakers.csv
│   │       ├── real_val_speakers.csv
│   │       └── real_test_speakers.csv
│   └── README.md
├── docs/
│   ├── data_and_training_design.md
│   ├── labeling_guideline.md
│   ├── dataset_schema.md
│   ├── experiment_log.md
│   └── decision_log.md
├── src/
├── configs/
├── results/
└── README.md
```

> 원본 음성 파일은 개인정보·동의·저작권 이슈가 있을 수 있으므로 GitHub에 직접 올리지 않는다. GitHub에는 메타데이터 형식, 데이터 수집·처리 절차, 재현 가능한 코드, 익명화된 예시만 올린다.

### 10.2 Manifest 필수 컬럼

```csv
sample_id,path,source,speaker_id,recording_id,start_sec,end_sec,task_label,energy_label,split,sample_rate,duration_sec,quality_flag
```

| 컬럼 | 설명 |
|---|---|
| `sample_id` | 고유 샘플 식별자 |
| `path` | 파일 경로 또는 안전한 상대 경로 |
| `source` | `real` 또는 `synthetic` |
| `speaker_id` | 화자 단위 분할을 위한 ID |
| `recording_id` | 원본 녹음 파일 ID |
| `start_sec`, `end_sec` | 구간 라벨인 경우의 시간 범위 |
| `task_label` | `normal`, `prolongation`, `tremor`, `energy_variation` 등 상위 라벨 |
| `energy_label` | `normal_energy`, `energy_fade_in`, `energy_fade_out`, `energy_ambiguous` 등 |
| `split` | `train`, `val`, `test` |
| `sample_rate` | 샘플레이트 |
| `duration_sec` | 분석 구간 길이 |
| `quality_flag` | 녹음 품질 또는 제외 사유 |

### 10.3 실험 로그 템플릿

| Experiment ID | Date | Git commit | Dataset version | Train real | Train synthetic | Batch ratio | Model | Seed | Val F1 | Test F1 | Notes |
|---|---|---|---|---:|---:|---|---|---:|---:|---:|---|
| E1 |  |  |  | 0 |  | 0:100 |  |  |  |  |  |
| E2 |  |  |  |  | 0 | 100:0 |  |  |  |  |  |
| E3 |  |  |  |  |  | random mix |  |  |  |  |  |
| E4 |  |  |  |  |  | 50:50 |  |  |  |  |  |

`Test F1`은 실험을 반복적으로 최적화하는 칸이 아니다. validation으로 최종안을 고른 뒤 마지막에 기록한다.

---

## 11. 학술제 발표를 위한 핵심 논리

### 11.1 문제 정의

- 실제 면접 음성의 연장, 떨림, 에너지변동은 수집·정밀 라벨링 비용이 높아 데이터가 부족하다.
- 합성 음성은 라벨을 정밀하게 부여할 수 있지만, 실제 음성과 완전히 동일하지 않다.
- 따라서 단일 출처에 의존하지 않고 실제와 합성의 장점을 결합해야 한다.

### 11.2 제안 방법

```text
실제 음성의 상위 라벨 확보
+ energy_variation 반자동 세분화
+ 복수 청취자 검증으로 고신뢰 fade 방향 라벨 구축
+ 합성 음성으로 세부 패턴 보완
+ 화자 독립 실제 평가로 일반화 성능 검증
```

### 11.3 발표에서 강조할 점

- 합성 데이터는 단순한 데이터 수 늘리기가 아니라, 명확한 방향 라벨을 공급한다.
- 실제 데이터는 서비스 적용 대상이므로 모델 학습·선택·평가에서 핵심 기준이다.
- 자동 라벨만 신뢰하지 않고 사람 청취 검증을 결합했다.
- 화자 단위 분할로 화자 누수를 방지했다.
- 최종 성능을 합성 test가 아니라 실제 test에서 평가했다.
- 비율 탐색을 무작정 크게 하지 않고, 데이터 부족 환경에 맞춰 전량 혼합 baseline과 최소 ablation으로 검증했다.

### 11.4 한 장 요약 도식

```text
[실제 일반인 음성]
normal / prolongation / tremor / energy_variation
                         │
                         ▼
         [에너지 궤적 자동 분석 + 블라인드 청취 검증]
                         │
                         ▼
       normal / fade_in / fade_out / ambiguous
                         │
                         ├─────────────┐
                         │             │
                         ▼             ▼
          [실제 train]          [실제 val / test]
                         ▲             │
                         │             │
[합성 normal / fade_in / fade_out]     │
                         │             │
                         ▼             ▼
                [공동학습 모델] ──→ [실제 일반화 성능 평가]
```

---

## 12. 현재 확정 사항과 미결 과제

### 12.1 현재 확정 사항

- 실제 일반인 음성에는 `normal`, `prolongation`, `tremor`, `energy_variation` 상위 라벨이 있다.
- 실제 `energy_variation`을 `fade_in`, `fade_out`, `ambiguous`로 세분화한다.
- 세분화는 자동 규칙만으로 확정하지 않으며, 사람 청취 검증을 거친다.
- 실제와 합성 데이터는 모두 train에 사용한다.
- 실제 validation/test에는 합성 데이터를 넣지 않는다.
- 실제 데이터는 화자 단위로 train/validation/test를 분리한다.
- 최종 성능은 실제 test 데이터에서만 판단한다.
- 초기에는 실제·합성 전량 혼합 baseline을 우선 구축한다.
- 합성이 과도하게 많은 경우 source-balanced batch를 비교한다.

### 12.2 미결 과제

- 실제 `energy_variation` 라벨이 파일 전체 단위인지, 시간 구간 단위인지 확인
- 각 라벨의 샘플 수, 화자 수, 화자별 클래스 분포 집계
- 녹음 sample rate, 채널 수, 음질, 동의 범위 정리
- 에너지 분석 프레임 길이, hop length, smoothing 방식 결정
- 실제 라벨링 파일럿을 통한 fade 판정 기준의 잠정화
- 청취자 수, 합의 규칙, 일치도 산출 방식 결정
- CNN 입력 특징(log-Mel, energy contour, F0 등)과 모델 구조 결정
- 실제·합성 sampling 전략의 E1~E4 실험 수행

---

## 13. 참고 문헌 및 근거

### 자동 운율 라벨링과 수동 검증

- Rosenberg, A. (2016). *An Automatic Prosody Tagger for Spontaneous Speech*. COLING 2016. 자동 운율 태깅에서 음향 특징을 사용하고 사람 주석과의 일치도를 평가한다.  
  URL: https://aclanthology.org/C16-1037.pdf

- Mertens, P. (2004). *The Prosogram: Semi-Automatic Transcription of Prosody*. Speech Prosody 2004. F0와 intensity 정보를 이용한 반자동 운율 분석 및 수동 전사 자료를 통한 검증을 제시한다.  
  URL: https://www.isca-archive.org/speechprosody_2004/mertens04_speechprosody.pdf

- Hirst, D. (2020). *Automatic Prosody Labelling and Assessment*. Oxford Research Encyclopedia of Linguistics. 자동 운율 라벨링의 목적과 방법론을 개관한다.  
  URL: https://academic.oup.com/edited-volume/34870/chapter/298318371

### 합성 데이터와 실제 데이터의 결합

- Apple Machine Learning Research. *Beyond Real Data: Synthetic Data through the Lens of Regularization*. 실제 데이터가 부족할 때 합성 데이터가 일반화에 도움을 줄 수 있으나, 과도한 합성 의존은 분포 불일치를 만들 수 있음을 논의한다.  
  URL: https://machinelearning.apple.com/research/beyond-real-data

- Ronchini, F., et al. (2024). *Synthetic Training Set Generation Using Text-to-Audio Models for Sound Event Detection*. DCASE 2024 Workshop. 실제 데이터와 텍스트-오디오 기반 합성 데이터를 결합한 학습 시나리오를 다룬다.  
  URL: https://dcase.community/documents/workshop2024/proceedings/DCASE2024Workshop_Ronchini_8.pdf

- Rossenbach, T., et al. (2023). *On the Relevance of Phoneme Duration Variability of Synthesized Speech for ASR*. ASRU 2023. 합성 음성을 저자원 또는 domain mismatch 상황의 음성 인식 개선에 활용할 수 있음을 다룬다.  
  URL: https://www-i6.informatik.rwth-aachen.de/publications/download/1249/Rossenbach-ASRU-2023.pdf

> 위 문헌은 본 프로젝트의 자동 라벨링 검증, 합성 데이터 활용, 실제 평가 원칙의 방법론적 배경으로 사용한다. 특정 dB 임계값이나 fade 방향 판정값을 직접 제공하는 문헌 근거는 현재 확보하지 않았으므로, 해당 기준은 실제 라벨링 파일럿 결과를 통해 프로젝트 내부의 잠정값으로 설정해야 한다.
