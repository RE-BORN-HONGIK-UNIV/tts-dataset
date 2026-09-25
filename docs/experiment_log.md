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


---

## 2026-09-25 — 연장(prolongation) 탐색 및 느린 정상 발화 비교

### 목적
1학기 실험에서 전체 발화속도가 느린 정상 음성이 연장으로 오탐되어 점수가 과도하게 낮아진 사례가 있었다.  
이에 따라 연장을 절대적인 지속시간만으로 판단하지 않고, 문장 안에서 주변 발화 단위보다 국소적으로 길어진 현상으로 다루기 위한 탐색을 진행했다.

### 사용 데이터
- 합성 연장 음성: 84개
  - 경로: `data/prolong/audio`
  - 수작업 텍스트 입력으로 생성한 TTS 연장 음성
- 실제 면접 연장 음성: 51개
  - 원본 경로: `C:\Users\seoyn\Documents\Reborn_Project_V2\01_Dataset_Raw\prolongation`
- 정상 TTS 원본: 71명 × 6문장 = 426개
  - 경로: `G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy`

### 수행한 탐색
- `librosa.pyin`을 사용해 WAV에서 유성 구간을 탐색했다.
- 합성 연장 84개에서 유성 구간 624개를 기록했다.
  - 결과: `data/prolong/analysis_preview/voiced_segments.csv`
- 실제 연장 51개에서 유성 구간 395개를 기록했다.
  - 결과: `data/prolong/analysis_preview/real_voiced_segments.csv`
- 동일 화자·동일 문장 정상 TTS 1개를 `rate=0.85`로 전체 시간 늘이기하여 느린 정상 음성 시험본을 생성했다.
  - 정상: 3.11초
  - 느린 정상: 3.66초
  - 청취 결과: 전체적으로 느려졌지만 특정 음절만 늘어난 연장처럼 들리지는 않음
- 정상 / 느린 정상 / 합성 연장 / 실제 연장을 파형과 스펙트로그램으로 비교했다.
  - 결과 그림: `data/prolong/analysis_preview/compare_normal_slow_synthetic_real.png`

### 관찰 결과
- 느린 정상은 여러 발화 구간이 전반적으로 함께 길어지는 모습이었다.
- 합성 및 실제 연장은 일부 유성 구간이 주변 구간보다 상대적으로 길게 이어지는 사례가 관찰되었다.
- 실제 연장 음성은 합성 연장 음성보다 유성 구간 길이와 총 유성 시간이 대체로 더 길었으나, 두 분포가 겹친다.
- 따라서 특정 지속시간 또는 단일 음향 지표만으로 연장을 판정하는 것은 부적절하다.

### 탐색 지표 결과
`최장 유성 구간 길이 ÷ 파일 내 유성 구간 길이 중앙값`을 계산했다.

| 파일 유형 | 비율 |
|---|---:|
| 정상 | 2.04배 |
| 느린 정상 | 2.26배 |
| 합성 연장 | 2.75배 |
| 실제 연장 | 2.38배 |

이 지표는 느린 정상과 실제 연장을 명확히 분리하지 못했다.  
따라서 이 비율은 최종 연장 판정 임계값으로 사용하지 않는다.

### 한계점
- `pyin`의 유성 구간은 음소 경계가 아니라 음성 신호 기반의 추정 구간이다.
- 유성 구간 길이는 여러 음절이 하나로 이어진 구간, 느린 발화, 호흡·녹음 조건의 영향도 함께 반영할 수 있다.
- 합성 연장과 실제 연장은 화자, 문장, 녹음 환경, TTS 여부가 달라 스펙트로그램 차이가 연장 현상만의 차이라고 볼 수 없다.
- 현재 생성한 느린 정상은 3개이며, `rate=0.85`는 학술적으로 확정된 연장 판정 또는 느린 발화 기준이 아니라 청취 검증용 잠정값이다.
- 스펙트로그램 기반 CNN이 연장 자체가 아니라 화자, TTS/실제 녹음 환경, 문장 또는 전체 말속도 차이를 지름길 특징으로 학습할 위험이 있다.

### 현재 결정
- 연장 모델은 에너지·떨림 모델과 분리된 독립 이진 분류기로 설계한다.
  - `normal(0)` vs `prolongation(1)`
- 정상 TTS 426개는 연장·떨림·에너지 모델에 공통 정상 원본으로 사용할 수 있다.
- 느린 정상 음성은 연장 오탐 방지를 위한 `normal(0)` 반례 데이터 후보로 유지한다.
- 지속시간의 절대 임계값 또는 `최장/중앙값` 비율만으로 연장을 판정하지 않는다.
- 실제 면접 연장 음성 51개는 합성 데이터와 비교하되, 최종 평가에 사용할 파일은 기준 조정에 사용한 파일과 분리한다.

### 다음 작업
1. 서로 다른 화자·문장을 포함하도록 느린 정상 시험본을 추가 생성하고 청취 검증한다.
2. 실제 연장 음성에서 청취상 연장 위치를 일부 수동 표시하여, 자동 유성 구간 탐색 결과와 비교한다.
3. 합성 연장 84개를 청취 기준으로 `사용 가능 / 애매함 / 제외`로 품질 점검한다.
4. 실제 음성 기반 검증 세트를 분리하고, 화자 단위로 train/validation/test 분할을 설계한다.