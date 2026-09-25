# Synthetic v1 실험 로그

## 문서 목적

이 문서는 `synthetic_v1` 데이터셋의 생성, 품질관리(QC), 주요 판단 및 결과를 기록한다.

본 데이터셋은 고립청년 대상 대면 면접 훈련 서비스의 1단계 음성 분석 개발을 위한 합성 음성 데이터셋이다. 여기서 생성한 에너지 fade 라벨은 음성 신호의 점진적 에너지 변화 특성을 기술하기 위한 합성 라벨이며, 실제 개인의 불안도, 정신건강 상태 또는 면접 수행 능력을 진단하거나 판정하는 라벨이 아니다.

---

## v1 데이터셋 개요

- 데이터셋 버전: `synthetic_v1`
- 기본 원본 클래스: `normal_energy`
- 파생 클래스: `energy_fade_in`, `energy_fade_out`
- 오디오 형식: WAV, 44.1 kHz, mono
- normal 원본 수: 426개
- energy_fade_in 파생본 수: 426개
- energy_fade_out 파생본 수: 426개
- 총 파생본 수: 852개
- 전체 WAV 수: 1,278개

### 클래스 의미

| 클래스 | 의미 |
|---|---|
| `normal_energy` | 에너지 fade 변조를 적용하지 않은 기준 음성 |
| `energy_fade_in` | 발화 초반보다 후반의 에너지가 상대적으로 증가하도록 합성 변조한 음성 |
| `energy_fade_out` | 발화 초반보다 후반의 에너지가 상대적으로 감소하도록 합성 변조한 음성 |

---

## 생성 절차

### 1. normal 음성 생성

- 실행 스크립트: `src/v1/generate_normal_v1.py`
- normal WAV 품질 점검 스크립트: `src/v1/qc_normal_wav.py`
- normal QC 요약 스크립트: `src/v1/summarize_normal_qc.py`
- clipping 후보 확인 스크립트: `src/v1/check_clipping_candidates.py`

normal 음성을 기준 파일로 사용하고, 파일별로 fade-in 및 fade-out 파생본을 생성했다.

### 2. energy fade 파생본 생성

- 실행 스크립트: `src/v1/generate_fade_variants.py`
- 입력: `audio/normal_energy/`
- 출력:
  - `audio/energy_fade_in/`
  - `audio/energy_fade_out/`
- strength manifest:
  - `metadata/fade_strength_manifest.csv`

각 normal 원본 WAV에 대해 `energy_fade_in` 및 `energy_fade_out` WAV를 각각 1개씩 생성했다. 원본-파생본은 파일명 규칙을 통해 1:1로 대응한다.

---

## Fade 변조 조건

### Strength 배정

- strength 범위: 12 dB 이상, 18 dB 미만
- strength strata 수: 6개
- strata별 파일 수: 각 71개
- 총 원본 수: 426개

| strength strata | 파일 수 |
|---|---:|
| 12–13 dB | 71개 |
| 13–14 dB | 71개 |
| 14–15 dB | 71개 |
| 15–16 dB | 71개 |
| 16–17 dB | 71개 |
| 17–18 dB | 71개 |

### 변조 방향

- `energy_fade_in`: 발화 후반부가 전반부보다 상대적으로 커지는 방향의 에너지 변화
- `energy_fade_out`: 발화 후반부가 전반부보다 상대적으로 작아지는 방향의 에너지 변화

---

## QC 방법

### 기술적 무결성 QC

- 실행 스크립트: `src/v1/qc_fade_variants.py`
- pair별 결과: `metadata/fade_qc_pairs.csv`
- strength 요약: `metadata/fade_qc_strength_summary.csv`
- 전체 요약: `metadata/fade_qc_summary.txt`

다음 항목을 검사했다.

- normal, fade-in, fade-out 파일 수
- 원본과 파생본의 1:1 대응 관계
- WAV 읽기 가능 여부
- sample rate: 44.1 kHz 여부
- 채널 수: mono 여부
- 원본과 파생본의 프레임 수 및 길이 일치 여부
- strength strata별 파일 수 및 strength 범위
- peak 값 경고 여부
- RMS 기반 전반·후반 에너지 방향성

### RMS 에너지 방향성 검사

RMS는 짧은 구간에서의 평균적인 소리 세기 또는 신호 에너지를 나타내는 지표로 사용했다. 전체 음성을 40 ms 분석 프레임과 20 ms hop으로 나누고, 각 프레임의 RMS 에너지를 dB로 변환했다.

- 분석 frame: 40 ms
- 분석 hop: 20 ms
- 비교 구간: 전체 발화의 전반 25%, 후반 25%
- 계산값: 후반 평균 RMS dB − 전반 평균 RMS dB

판정 원칙은 다음과 같다.

| 클래스 | 단독 파생본 검사에서 기대한 방향 |
|---|---|
| `energy_fade_in` | 후반 − 전반 에너지 차이가 0 dB보다 커야 함 |
| `energy_fade_out` | 후반 − 전반 에너지 차이가 0 dB보다 작아야 함 |

이 기준은 합성 fade의 기술적 방향성을 확인하기 위한 운영 기준이다. 실제 면접 불안도 또는 실제 화자의 에너지변동을 판정하는 임상·심리 임계값은 아니다.

---

## QC 결과

### 1차 fade QC

| 항목 | 결과 | 판정 |
|---|---:|---|
| normal 파일 수 | 426 / 426 | PASS |
| energy_fade_in 파일 수 | 426 / 426 | PASS |
| energy_fade_out 파일 수 | 426 / 426 | PASS |
| 검사 pair 수 | 852 / 852 | PASS |
| pair PASS | 841 | PASS |
| pair REVIEW | 11 | 재검증 |
| pair FAIL | 0 | PASS |
| strength strata PASS | 6 / 6 | PASS |
| strength strata FAIL | 0 | PASS |

1차 QC에서 기록된 REVIEW 항목은 다음과 같다.

- `fade_in_nonpositive_delta`: 8개
- `fade_out_nonnegative_delta`: 3개

이 11개는 파생 WAV만 단독으로 평가했을 때 전반·후반 RMS 방향이 기대 부호와 일치하지 않았던 파일이다. 이는 원본 문장의 고유 억양, 문장 시작·종결 강세 또는 발화 구간별 에너지 분포가 전반·후반 평균에 영향을 주었을 가능성을 고려하여 원본 대비 재검증을 수행했다.

### REVIEW 원본 대비 재검증

- 실행 스크립트: `src/v1/summarize_fade_qc_reviews.py`
- REVIEW pair 수: 11개
- 원본 대비 기대 방향 shift 충족: 11개
- 원본 대비 기대 방향 shift 미충족: 0개
- 최종 재생성 대상: 없음

원본 대비 변화량은 다음과 같이 계산했다.

```text
energy_delta_shift_db
= 파생본의 전후반 에너지 차이
− 원본의 전후반 에너지 차이
```

판정 원칙은 다음과 같다.

| 클래스 | 원본 대비 기대 방향 |
|---|---|
| `energy_fade_in` | `energy_delta_shift_db > 0` |
| `energy_fade_out` | `energy_delta_shift_db < 0` |

재검증 결과, 1차 REVIEW 11개는 모두 원본 대비 의도한 fade 방향으로 변화했다.

- fade-in REVIEW 8개: 원본 대비 +10.875 dB ~ +15.610 dB 변화
- fade-out REVIEW 3개: 원본 대비 −11.685 dB ~ −14.631 dB 변화

따라서 11개 파일은 오류가 아니라 원본 문장 억양의 영향을 받은 자동 QC의 경계 사례로 판단했으며, 전체 852개 파생본을 유지했다.

---

## 최종 판정

`synthetic_v1`의 energy fade 파생본은 기술적 QC를 통과했다.

- 파일 수, 클래스별 분포, 원본-파생본 대응 관계가 정상이다.
- 파일 읽기, sample rate, 채널 수, 원본-파생본 길이 검사에서 FAIL이 없다.
- strength strata 6개가 각각 71개씩 균형 있게 구성되었다.
- 1차 RMS 방향성 QC에서 발생한 REVIEW 11개는 원본 대비 재검증에서 모두 의도한 fade 방향을 충족했다.
- 최종 FAIL은 0개이며, 재생성 대상은 없다.

---

## 한계와 후속 작업

본 QC는 파형 수준의 기술적 무결성과 원본 대비 에너지 방향성을 점검한 결과다. 다음 사항은 별도 검토가 필요하다.

- 사람이 들었을 때 fade 변화가 자연스러운지에 대한 청취 평가
- fade 라벨이 실제 면접 상황에서 나타나는 불안 음성 특성과 어느 정도 유사한지에 대한 타당화
- 실제 사용자 음성에 적용할 때의 마이크, 환경 소음, 발화 길이 차이에 대한 강건성 검토
- 에너지 특성 하나만으로 실제 불안도를 단정하지 않도록 시선·표정·언어적 특성과 결합한 다중지표 설계

권장 후속 작업은 각 클래스에서 표본을 무작위로 10개씩 추출하여 총 30개를 블라인드 청취하고, 자연스러움, fade 방향 인지 가능 여부, 왜곡·클리핑 여부를 기록하는 것이다.