# Interview Speech Disfluency Detection

> 고립청년 대상 대면 면접훈련 서비스의 음성 분석 연구 모듈

면접 연습 중 나타나는 음성 비유창성과 발화 특성을 분석하여, 사용자가 자신의 말하기 패턴을 돌아보고 훈련할 수 있도록 돕는 연구 프로젝트입니다.

이 저장소는 실제 일반인 음성과 합성 TTS 음성을 결합하여, 면접 발화에서 나타나는 음향적 특성을 탐지하는 연구 흐름을 관리합니다.

```text
실제 일반인 음성 라벨
+ 합성 TTS normal 원본 생성
→ normal 원본에서 fade-in / fade-out 파생 음성 생성
→ 실제 energy_variation의 반자동 세분화 및 청취 검증
→ 실제·합성 음성 공동학습
→ 화자 독립 실제 음성 평가
```

---

## 1. 프로젝트 목표

본 프로젝트는 고립청년의 대면 면접 준비를 지원하는 훈련 서비스의 1단계 음성 분석 모듈을 개발합니다.

분석 대상은 다음과 같습니다.

- 채움말(filler)
- 멈춤(pause)
- 연장(prolongation)
- 떨림(tremor)
- 에너지변동(energy variation)

채움말과 멈춤은 규칙 기반 분석을 우선 적용합니다. 연장, 떨림, 에너지변동은 음향 특징 기반 CNN 분류 모델로 탐지합니다.

> 이 모델은 의학적·임상적 진단을 수행하지 않습니다. 면접 훈련 상황에서 관찰되는 말하기 특성을 사용자에게 전달하는 보조 피드백 도구입니다.

---

## 2. 연구 질문

본 프로젝트는 다음 질문에 답하는 것을 목표로 합니다.

1. 실제 음성 데이터가 적은 환경에서 합성 음성이 음성 비유창성 탐지에 도움이 되는가?
2. 실제 `energy_variation` 구간을 `energy_fade_in`과 `energy_fade_out`으로 반자동 세분화할 수 있는가?
3. 합성·실제 데이터를 함께 학습할 때, 화자 독립 실제 음성 평가에서 일반화 성능이 개선되는가?
4. 실제 음성과 합성 음성의 비율 및 sampling 방식이 모델 성능에 어떤 영향을 주는가?

---

## 3. 데이터와 라벨

### 3.1 실제 일반인 음성

실제 일반인 음성은 모델이 최종적으로 일반화해야 하는 목표 도메인입니다. 현재 데이터는 화자 ID(`speaker_id`)와 다음 상위 라벨을 포함합니다.

```text
normal
prolongation
tremor
energy_variation
```

`energy_variation`은 후속 반자동 라벨링과 사람 청취 검증을 통해 아래 방향 라벨로 확장합니다.

```text
normal_energy
energy_fade_in
energy_fade_out
energy_ambiguous
recording_issue
```

| 라벨 | 설명 | 초기 3클래스 에너지 모델 사용 여부 |
|---|---|---|
| `normal_energy` | 뚜렷한 단조 에너지 상승·하강이 없는 발화 | 사용 |
| `energy_fade_in` | 발화 전반부보다 후반부 에너지가 전반적으로 증가 | 사용 |
| `energy_fade_out` | 발화 전반부보다 후반부 에너지가 전반적으로 감소 | 사용 |
| `energy_ambiguous` | 변화는 있으나 방향을 신뢰성 있게 판단하기 어려움 | 제외·보류 |
| `recording_issue` | 잡음, clipping, 마이크 변화 등으로 판단이 어려움 | 제외 |

### 3.2 합성 TTS 음성

합성 데이터는 부족한 실제 데이터를 보완하고, 명확한 에너지 변화 방향을 가진 사례를 공급하기 위해 생성합니다.

```text
normal_energy
energy_fade_in
energy_fade_out
```

합성 `energy_fade_in`과 `energy_fade_out`은 normal 원본의 진폭 포락선을 점진적으로 조절해 만드는 신호 처리 기반 파생 라벨입니다. 따라서 이 라벨은 실제 화자의 정서, 불안도 또는 자연발화의 비유창성을 뜻하지 않습니다.

합성 데이터는 방향성 있는 음향 패턴을 학습시키기 위한 보조 자료로만 사용하며, 실제 데이터에 대한 성능 평가는 별도로 수행합니다.

### 3.3 synthetic_v1 상태

`synthetic_v1`은 71개 합성 화자와 6개 공통 면접 문장을 조합하여 구성했습니다.

```text
71 speakers × 6 sentences = 426 normal WAV files
```

| 클래스 | 파일 수 | 역할 |
|---|---:|---|
| `normal_energy` | 426 | 에너지 변조가 없는 기준 원본 |
| `energy_fade_in` | 426 | 후반부 에너지가 상대적으로 증가하는 합성 파생본 |
| `energy_fade_out` | 426 | 후반부 에너지가 상대적으로 감소하는 합성 파생본 |
| 합계 | 1,278 | synthetic_v1 전체 WAV |

`synthetic_v1`의 normal 원본과 fade 파생본은 파일 수, 원본-파생본 대응, 오디오 형식, 길이 일치, 강도 분포 및 원본 대비 에너지 방향성 QC를 완료했습니다. 최종 재생성 대상은 없습니다.

생성 조건, QC 방법, 상세 결과, 예외 검토와 최종 판정은 [`docs/experiment_log.md`](docs/experiment_log.md)에 기록합니다.

---

## 4. 에너지변동 세분화

이 절에서 `fade_in`, `fade_out`, `ambiguous`는 에너지 변화의 방향 개념을 뜻합니다. 데이터셋의 최종 라벨명은 각각 `energy_fade_in`, `energy_fade_out`, `energy_ambiguous`를 사용합니다.

실제 데이터의 `energy_variation`은 방향 라벨이 없으므로, 다음 반자동 절차를 거쳐 `energy_fade_in`, `energy_fade_out`, `energy_ambiguous`로 세분화합니다.

```text
기존 energy_variation 구간
→ VAD 기반 유성 구간 확인
→ RMS/log-energy 궤적 추출 및 평활화
→ 전반부·후반부 에너지 차이 및 회귀 기울기 계산
→ fade_in / fade_out / ambiguous 후보 생성
→ 복수 청취자의 블라인드 검증
→ 고신뢰 최종 라벨 확정
```

자동 분석은 최종 정답을 생성하는 도구가 아니라, 사람이 검토할 후보를 효율적으로 찾는 도구입니다.

- 최소 2명, 가능하면 3명의 청취자가 독립적으로 판단합니다.
- 자동 후보 라벨은 청취자에게 사전에 제공하지 않습니다.
- 의견이 일치하지 않거나 비단조적인 변화는 `energy_ambiguous`로 보류할 수 있습니다.
- `recording_issue`는 별도 관리하고 학습·평가에서 제외합니다.

### 잠정 판정 정의

에너지 추세는 아래 값으로 계산합니다.

\[
\Delta E_{dB} = \overline{E}_{\text{후반부}} - \overline{E}_{\text{전반부}}
\]

\[
E(t) = \beta_0 + \beta_1 t + \epsilon
\]

- \(\Delta E_{dB} > 0\), \(\beta_1 > 0\): `energy_fade_in` 후보
- \(\Delta E_{dB} < 0\), \(\beta_1 < 0\): `energy_fade_out` 후보
- 방향이 약하거나 지표 간 방향이 불일치: `energy_ambiguous` 후보

> 특정 dB 차이 또는 기울기 값은 아직 보편적·학술적으로 확정된 실제 발화 판정 임계값이 아닙니다. 실제 음성 라벨링 파일럿에서 사람 청취 결과와 음향 특징을 비교하여 프로젝트 내부의 잠정 기준을 설정합니다.

---

## 5. 학습과 평가 원칙

### 5.1 실제·합성 공동학습

데이터가 부족하므로 실제와 합성 데이터를 모두 학습에 사용합니다.

```text
Training data
= 고신뢰 실제 train 데이터
+ 합성 normal_energy / energy_fade_in / energy_fade_out 데이터
```

권장 학습 흐름은 다음과 같습니다.

```text
1. 합성 데이터 사전학습
2. 합성 + 실제 train 데이터 공동학습
3. 필요 시 실제 train 데이터만으로 저학습률 fine-tuning
4. 실제 validation 성능으로 모델·threshold 선택
5. 실제 test에서 최종 성능 평가
```

### 5.2 화자 독립 데이터 분할

실제 데이터는 파일 단위가 아니라 **화자 단위**로 분할합니다.

```text
Real train       : 학습용 실제 화자
Real validation  : 모델 선택용 실제 화자
Real test        : 최종 평가용 실제 화자
Synthetic train  : 합성 데이터
```

같은 화자의 발화가 train과 validation/test에 겹치지 않도록 합니다. 이는 모델이 특정 화자의 음색, 발화 습관 또는 마이크 조건을 외워 성능이 과대평가되는 data leakage를 막기 위함입니다.

### 5.3 평가 데이터 원칙

```text
합성 데이터: train에만 사용
실제 validation: 모델 선택과 threshold 결정에 사용
실제 test: 최종 성능 보고에만 사용
```

최종 성능은 실제 test 화자 데이터에서 측정합니다.

### 5.4 초기 에너지 모델

초기 에너지 방향 분류 모델은 다음 3개 클래스를 출력합니다.

```text
normal_energy
energy_fade_in
energy_fade_out
```

실제 데이터에 방향 라벨이 아직 충분하지 않을 때는 아래와 같이 이진 성능도 평가합니다.

\[
P(\text{energy_variation})
=
P(\text{energy_fade_in})
+
P(\text{energy_fade_out})
\]

```text
normal_energy vs energy_variation
```

`energy_fade_in`과 `energy_fade_out`의 방향별 정확도는 실제 방향 라벨이 사람 검증을 통해 확보된 후에만 보고합니다.

---

## 6. 최소 실험 설계

| ID | 학습 데이터 | 목적 |
|---|---|---|
| E1 | 합성 데이터만 | 합성 학습 패턴의 실제 음성 전이 성능 확인 |
| E2 | 실제 데이터만 | 소량 실제 데이터 기준선 확인 |
| E3 | 실제 + 합성 전량 혼합 | 기본 모델 및 전량 활용 전략 확인 |
| E4 | 실제 + 합성 + source-balanced batch | 합성 과대표집 영향 확인 |

모든 실험은 동일한 실제 validation/test 화자 분할에서 비교합니다.

- 실험 설정 선택은 실제 validation 성능을 기준으로 합니다.
- 최종안이 결정된 뒤 실제 test를 사용합니다.
- test 결과를 보고 모델 구조나 threshold를 반복적으로 변경하지 않습니다.
- 실제·합성 비율은 validation 성능을 바탕으로 조정합니다.

### 초기 source balance

합성 데이터가 실제 데이터보다 많은 경우, 실제 데이터가 업데이트 과정에서 묻히지 않도록 source-balanced mini-batch를 비교합니다.

```text
잠정 운영값: 실제 50% + 합성 50% per mini-batch
```

이 1:1 비율은 학술적 최적값이 아니라 합성 데이터 과대표집을 막기 위한 잠정 운영값입니다.

```text
source balance: 실제 / 합성 비율
class balance: normal_energy / energy_fade_in / energy_fade_out 비율
```

---

## 7. 평가 지표

에너지변동 탐지 및 방향 분류에서 다음 지표를 사용합니다.

- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC
- Confusion matrix

방향 라벨이 없는 실제 데이터에서는 `normal_energy` 대 `energy_variation` 이진 성능을 우선 보고합니다.

---

## 8. 저장소 구조

```text
.
├── README.md
├── docs/
│   ├── data_and_training_design.md
│   ├── labeling_guideline.md
│   ├── dataset_schema.md
│   ├── experiment_log.md
│   └── decision_log.md
├── src/
│   ├── pilots/
│   │   ├── s1_check_voices.py
│   │   ├── s2_generate_normal_pilot.py
│   │   ├── s3_generate_energy.py
│   │   ├── s3_screen_speakers.py
│   │   ├── s4_build_master_metadata.py
│   │   ├── s4_generate_energy_pilot.py
│   │   └── s4_qc_energy_pilot.py
│   └── v1/
│       ├── generate_normal_v1.py
│       ├── qc_normal_wav.py
│       ├── summarize_normal_qc.py
│       ├── check_clipping_candidates.py
│       ├── generate_fade_variants.py
│       ├── qc_fade_variants.py
│       └── summarize_fade_qc_reviews.py
├── configs/
├── data/
│   ├── README.md
│   ├── metadata/
│   └── sample/
├── notebooks/
├── results/
├── requirements.txt
└── .gitignore
```

생성된 전체 WAV와 상세 생성·QC 결과 파일은 용량 및 데이터 관리 정책상 Git에서 제외합니다. 코드가 참조하는 데이터셋 루트는 로컬 또는 Drive의 `synthetic_v1`이며, 공개 저장소에는 비식별 예시, 스키마, 코드, 문서 및 집계 결과만 포함합니다.

### 데이터 보안

실제 원본 음성은 GitHub에 업로드하지 않습니다. 실제 음성에는 화자 식별 가능성이 있을 수 있으므로 동의 범위, 접근 권한 및 보관 정책을 별도로 관리해야 합니다.

GitHub에는 다음만 포함합니다.

- 데이터셋 구조와 익명화된 metadata 예시
- 데이터 생성·전처리·학습·평가 코드
- 문서와 설정 파일
- 집계된 실험 결과
- 공개 가능한 합성 샘플 또는 비식별 예시

`.gitignore`에는 원본 음성, 모델 가중치 및 개별 실험 산출물을 포함합니다.

```gitignore
# Private or raw audio
data/raw/
data/private/
*.wav
*.mp3
*.flac
*.m4a

# Models and experiment outputs
models/
checkpoints/
runs/
wandb/
mlruns/

# Local environment
.venv/
venv/
.env
__pycache__/
*.py[cod]
.ipynb_checkpoints/
```

---

## 9. 문서

| 문서 | 역할 |
|---|---|
| [데이터·라벨링·학습·평가 설계서](docs/data_and_training_design.md) | 연구 데이터, 라벨, 학습 및 평가 설계의 상세 정의 |
| [라벨링 가이드라인](docs/labeling_guideline.md) | 실제 음성 청취 검증과 라벨 부여 절차 |
| [데이터셋 스키마](docs/dataset_schema.md) | 파일명, metadata 컬럼, 라벨 및 데이터 구조 정의 |
| [실험 로그](docs/experiment_log.md) | 생성, QC, 실행 결과, 예외 검토 및 변경 이력 |
| [의사결정 기록](docs/decision_log.md) | 주요 설계 선택, 근거, 대안 및 변경 결정 |

---

## 10. 발표용 핵심 메시지

```text
실제 음성 데이터는 적고 세부 방향 라벨이 제한적입니다.

따라서,
1. 실제 energy_variation을 자동 음향 분석과 사람 청취 검증으로 세분화하고,
2. 합성 음성으로 방향이 명확한 사례를 보완하며,
3. 실제·합성 데이터를 함께 학습하고,
4. 화자 독립 실제 음성 평가로 서비스 적용 가능성을 검증합니다.
```

---

## 11. 참고 문헌

### 자동 운율 라벨링과 수동 검증

- Rosenberg, A. (2016). *An Automatic Prosody Tagger for Spontaneous Speech*. COLING 2016.  
  [PDF](https://aclanthology.org/C16-1037.pdf)

- Mertens, P. (2004). *The Prosogram: Semi-Automatic Transcription of Prosody*. Speech Prosody 2004.  
  [PDF](https://www.isca-archive.org/speechprosody_2004/mertens04_speechprosody.pdf)

- Hirst, D. (2020). *Automatic Prosody Labelling and Assessment*. Oxford Research Encyclopedia of Linguistics.  
  [Link](https://academic.oup.com/edited-volume/34870/chapter/298318371)

### 합성·실제 데이터 결합

- Apple Machine Learning Research. *Beyond Real Data: Synthetic Data through the Lens of Regularization*.  
  [Link](https://machinelearning.apple.com/research/beyond-real-data)

- Ronchini, F., et al. (2024). *Synthetic Training Set Generation Using Text-to-Audio Models for Sound Event Detection*. DCASE 2024 Workshop.  
  [PDF](https://dcase.community/documents/workshop2024/proceedings/DCASE2024Workshop_Ronchini_8.pdf)

- Rossenbach, T., et al. (2023). *On the Relevance of Phoneme Duration Variability of Synthesized Speech for ASR*. ASRU 2023.  
  [PDF](https://www-i6.informatik.rwth-aachen.de/publications/download/1249/Rossenbach-ASRU-2023.pdf)

---

## License

라이선스 및 데이터 공개 범위는 실제 음성 데이터의 동의 조건, 사용 권한 및 팀의 배포 정책을 확인한 뒤 결정합니다.