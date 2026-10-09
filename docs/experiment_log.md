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

---

## 2026-09-26 — 느린 정상 발화와 실제 연장 구간 확대 비교

### 수행
- 실제 면접 음성 `pr_14.wav`를 듣고, 단어 끝의 한 소리를 길게 끄는 구간을 대략 1.8~2.5초로 표시했다.
- 동일한 합성 발화의 정상 TTS와 느린 정상 시험본(`rate=0.85`), 그리고 별개의 실제 면접 음성 `pr_14.wav`에서 청취로 표시한 구간을 파형·스펙트로그램으로 확대해 나란히 살펴봤다.
- 탐색 코드: `compare_local_prolong.py`
- 결과 그림: `data/prolong/analysis_preview/local_prolong_comparison.png`

### 관찰
- 정상 TTS와 느린 정상에서는 확대 구간 안에서도 여러 소리의 전환이 보였다.
- `pr_14.wav`의 청취상 연장 구간에서는 한 소리의 주기적인 성분이 비교적 오래 이어지는 모습이 보였다.
- 그러나 `pr_14.wav`에서 들은 연장 구간(약 1.8~2.5초)은 이 파일의 **최장 유성 구간이 아니었다**. 최장 구간만 찾는 방식은 실제 연장을 놓칠 수 있다.

### 한계 및 다음 확인
- 실제 음성과 TTS는 화자·문장·녹음 조건이 달라, 그림의 차이를 연장 현상만의 차이로 해석할 수 없다.
- 청취 후 연장 위치를 골라 확대했으므로, 이 그림은 **자동 탐지 성능 검증이 아니다**.
- 현재는 실제 연장 1개 사례의 관찰이다. 여러 화자의 느린 정상과 실제 연장에 같은 비교를 반복해야 한다.
- 이 관찰을 근거로 지속시간·영상 패턴의 판정 임계값을 정하지 않는다.


### 추가 탐색 — 화자별 느린 정상과 실제 연장 후보

- 서로 다른 합성 화자 `spkS001`, `spkS002`, `spkS003`의 정상 음성을 각각 `rate=0.85`로 느리게 만든 시험본을 청취했다. `rate=0.85`는 탐색용 잠정 설정이지 연장 판정 기준이 아니다.
- `spkS001`과 `spkS002`는 특정 소리가 두드러지게 끌리기보다 전체적으로 느려진 것으로 들렸다.
- `spkS003`은 약 1초 후반의 “안녕하세여어~”가 두드러졌지만, 원본에도 비슷한 문장 끝 억양이 있었다. 따라서 속도 변환으로 새로 생긴 연장으로 판정하지 않고 보류했다.
- 실제 음성 `data/prolong/audio/prolong_spk009_f_001.wav`에서는 시작~약 1초에 “이버어어언~ 방학에는”으로 들리는 구간을 연장 후보로 표시했다. 정확한 경계와 최종 라벨은 미확정이다.
- 추가 그림: `data/prolong/analysis_preview/local_prolong_spk009_comparison.png`
- 추가 그림의 위 두 행은 `spkS001`의 정상 원본(1.0–2.5초)과 그 느린 버전(1.18–2.94초)이며, 아래 행은 별개의 실제 음성 `prolong_spk009_f_001.wav`(0.0–1.2초)이다. 음소·화자·시간축을 정렬한 1:1 비교가 아니다.
- 실제 음성의 약 0.35–0.70초에 이어지는 음향 패턴이 관찰되지만, 이를 “버” 소리의 정확한 경계나 자동 연장 판정으로 확정하지 않는다.
- 다음 단계: 실제 연장 후보(`pr_14`, `prolong_spk009`)와 느린 정상·문장 끝 억양(`spkS003`)의 관찰을 참고하여, 특정 소리만 이어지는 연장 TTS 시험본을 만들고 청취로 확인한다. 현재 그림만으로 연장 판정 임계값을 정하지 않는다.

### 연장 표기 TTS 파일럿 — sent_06

- 정상 원문: `이번 기회를 통해 더 좋은 결과를 만들고 싶습니다.`
- 같은 Typecast 화자·문장 설정에서 `이번`의 연장 표기 후보 A(`이버언`), B(`이버어언`)를 시험했다.
- `spkS001` 청취: A는 정상과 차이가 약했고, B는 주변 소리보다 끄는 소리가 뚜렷하며 한 소리가 이어지는 느낌이었다.
- `spkS002` 청취: B에서 한 소리가 자연스럽게 이어지는 것으로 들렸다.
- 생성 파일은 `data/prolong/text_pilot/`에, 원문·변형 텍스트·화자·파일명은 같은 폴더의 `manifest.csv`에 저장했다.
- 두 화자의 한 문장에 대한 청취 결과이며, 다른 단어·문장에 적용 가능한 생성 규칙이나 학습용 라벨로 확정하지 않는다.

### 공통 정상 원문과 연장 위치 파일럿

아래 6개 정상 원문을 유지하고, 각 문장에서 연장할 위치를 달리하여 시험한다.
1차 시험은 `spkS001`에서 완료했으며, 2차 시험은 같은 문장 안에서
1차 위치와 겹치지 않는 다른 단어를 대상으로 진행할 예정이다.

| 문장 ID | 정상 원문 |
|---|---|
| sent_01 | 안녕하세요, 저는 새로운 일을 성실하게 배우고 싶습니다. |
| sent_02 | 어려운 문제가 생기면 우선 상황을 살펴보겠습니다. |
| sent_03 | 저는 오늘 맡은 일을 세심하게 마무리하겠습니다. |
| sent_04 | 서로의 의견을 듣고 제 생각을 분명하게 말하겠습니다. |
| sent_05 | 새로운 환경에서도 차분한 마음으로 적응하겠습니다. |
| sent_06 | 이번 기회를 통해 더 좋은 결과를 만들고 싶습니다. |

#### 1차: 처음 선택한 위치 — spkS001 시험 완료

각 문장의 아래 단어 한 곳을 연장해 A/B/C를 청취 비교했다.
A는 정상본과 차이가 작았고, B는 여섯 문장에서 연장이 들리며 자연스럽게
이어졌다. C는 더 길게 들렸지만 sent_02에서는 음이 분리되어 들렸다.

| 문장 ID | 1차 연장 위치 | 사용하기로 한 표기 | 선택 |
|---|---|---|---|
| sent_01 | 배우고 | 배우우우우고 | C |
| sent_02 | 살펴보겠습니다 | 살펴보오오겠습니다 | B |
| sent_03 | 마무리하겠습니다 | 마무우우우리하겠습니다 | C |
| sent_04 | 말하겠습니다 | 말하아아아겠습니다 | C |
| sent_05 | 차분한 | 차아아아분한 | C |
| sent_06 | 이번 | 이버어어언 | C |

- sent_02 C(`살펴보오오오겠습니다`)는 `보오 / 오겠습니다`처럼
  끊겨 들려 사용하지 않는다.
- 나머지 다섯 문장의 C는 B보다 연장이 분명하고 청취상 자연스럽게 이어져
  1차 위치의 후보로 선택했다.
- 이 선택은 `spkS001`에 대한 결과다. 모든 화자에게 같은 표기가
  통과한다는 뜻은 아니다.

#### 2차: 겹치지 않는 새 위치 — spkS001 시험 완료

정상 원문은 유지하고, 각 문장에서 1차와 다른 단어 한 곳을 골라
B/C 표기로 생성·청취했다. 생성 파일은 `data/prolong/text_pilot/`에,
2차 입력 텍스트와 파일 정보는 `alt2_manifest.csv`에 기록했다.

| 문장 ID | 1차 위치 | 2차 위치 | 2차 B/C 청취 결과 | 결정 |
|---|---|---|---|---|
| sent_01 | 배우고 | 새로운 | B/C 모두 끊기지는 않지만 `새로오온~`처럼 들려 자연스러움이 애매함 | B/C 모두 보류 |
| sent_02 | 살펴보겠습니다 | 우선 | B/C 모두 정상본과 구별되고 자연스럽게 이어짐 | B/C 모두 통과 |
| sent_03 | 마무리하겠습니다 | 오늘 | B/C 모두 정상본과 구별되고 자연스럽게 이어짐 | B/C 모두 통과 |
| sent_04 | 말하겠습니다 | 서로의 | B/C 모두 정상본과 구별되고 자연스럽게 이어짐 | B/C 모두 통과 |
| sent_05 | 차분한 | 마음으로 | B/C 모두 정상본과 구별되고 자연스럽게 이어짐 | B/C 모두 통과 |
| sent_06 | 이번 | 좋은 | B/C 모두 정상본과 구별되고 자연스럽게 이어짐 | B/C 모두 통과 |

- 통과한 B와 C는 연장 길이가 서로 달라도 **둘 다 보존**한다.
  길이를 하나로 고정하거나 B/C 중 반드시 하나만 선택하지 않는다.
- `sent_01`의 2차 위치는 끊김이 없더라도 자연스러움이 애매하므로,
  현재 연장 학습 후보로 확정하지 않는다. 1차 위치 `배우고`의
  선택본 C는 그대로 유지한다.
- 이 결과는 `spkS001`의 2차 위치 파일럿에 대한 청취 판단이다.
  다른 화자에서도 같은 표기가 통과하는지는 아직 확인하지 않았다.

  #### 3차: 다른 화자 적용 파일럿 

1·2차 시험은 주로 `spkS001`에서 진행했으므로, 같은 연장 표기가
다른 화자에게도 자연스럽게 적용되는지 확인하기 위해 승인 화자
`spkS006`(Munsu), `spkS009`(Seungyeon), `spkS010`(Sangwoo)를
추가 시험했다.

- 생성 계획: 화자당 1차 위치 6개 + 2차 위치 5개 = 11개,
  세 화자 총 33개 후보
- 1차 위치: `spkS001`에서 잠정 선택한 표기를 사용
  (`sent_02`는 B, 나머지 문장은 C)
- 2차 위치: `sent_02`~`sent_06`에서 화자·문장에 따라 B 또는 C를 배정
- 2차 `sent_01`의 `새로운`은 앞선 시험에서 자연스러움이 애매해 제외
- 생성 코드: `generate_text_prolong_multispeaker_pilot.py`
- 계획 출력에서 `새 생성 대상: 33개` 확인 후 생성 작업을 진행했다.
- 이 단계는 71명 전체 대량 생성이 아니라 다화자 파일럿이다.

##### 청취 중 눈에 띈 파일

아래는 우선 보고된 6개 파일의 청취 메모다. 이 목록에 없는 파일을
자동으로 통과 처리하지 않는다.

| 파일 | 청취 메모 | 현재 QC 상태 |
|---|---|---|
| `spkS006__sent_03__target_02__C.wav` | `오 오늘`로 들림. 메모의 `끊김 0`이 ‘없음’인지 ‘O(있음)’인지 확인 필요 | 재확인 |
| `spkS006__sent_04__target_02__B.wav` | `서.어.로`처럼 분리되어 들리는지 애매함 | 보류 |
| `spkS009__sent_04__target_02__C.wav` | `서.어.로`처럼 분리되어 들리는지 애매함 | 보류 |
| `spkS009__sent_06__target_02__C.wav` | `조오오은`의 연결이 애매함 | 보류 |
| `spkS010__sent_03__target_02__C.wav` | 높낮이 변화를 주며 길게 끌다가 끊겨 들림. 이미지가 연장처럼 보일 가능성과 별개로 청취상 연결성이 부족함 | 이번 연장 후보에서 제외 |
| `spkS010__sent_04__target_01__C.wav` | `말 하아아~ 아~ 아~아`처럼 여러 번 분리되어 들림. 합성 오류처럼 느껴짐 | 이번 연장 후보에서 제외 |

##### 현재 판단과 다음 작업

- `spkS001`에서 통과한 텍스트 표기가 다른 화자에게 항상 자연스럽게
  적용되지는 않았다. 따라서 현재 표기를 71명 전체에 그대로
  일괄 적용하지 않는다.
- 높낮이가 조금 변하는 자연스러운 연장과, 한 음이 끊겨 여러 번
  발화되는 결과를 구분해 청취 검수한다.
- 애매한 파일은 `통과`로 확정하지 않는다. 분리되어 들리는 파일은
  그림만 보고 연장 데이터로 되살리지 않는다.
- 다음 작업 시 위 첫 번째 파일의 `끊김 0` 의미를 확인하고,
  아직 결과를 기록하지 않은 파일을 `통과 / 보류 / 제외`로 검수한다.
- 검수 결과를 모은 뒤 화자·문장·목표 위치별로 B/C 유지, 다른
  표기 재시험, 해당 위치 제외 중 무엇이 적절한지 결정한다.
- 추가 Typecast 생성과 71명 대량 생성은 그 결정 전까지 보류한다.


## 2026-10-02 — 연장 표기 TTS 다화자 파일럿: spkS006 청취 QC

### 대상
- 화자: `spkS006`
- 대상 파일: 11개
- 목적: `spkS001`에서 선택한 연장 표기가 다른 화자에서도
  자연스럽고 연속적인 연장으로 합성되는지 청취로 확인한다.
- 판정: `통과 / 보류 / 제외`
- 주의: 이 청취 판정은 합성 데이터의 품질관리용이며, 실제 개인의
  불안도나 임상적 연장을 진단·판정하는 기준이 아니다.

### 결과

- 통과: 8개
  - `spkS006__sent_01__target_01__C.wav`
  - `spkS006__sent_02__target_01__B.wav`
  - `spkS006__sent_02__target_02__B.wav`
  - `spkS006__sent_03__target_01__C.wav`
  - `spkS006__sent_04__target_01__C.wav`
  - `spkS006__sent_05__target_01__C.wav`
  - `spkS006__sent_05__target_02__C.wav`
  - `spkS006__sent_06__target_01__C.wav`

- 제외: 3개
  - `spkS006__sent_03__target_02__C.wav`
    - 사유: “오, 오늘 맡은~”처럼 목표 위치가 분리되어 들림.
  - `spkS006__sent_04__target_02__B.wav`
    - 사유: “서.어.로의~”처럼 목표 위치가 분리되어 들림.
  - `spkS006__sent_06__target_02__B.wav`
    - 사유: 발화는 연속적이나 목표 연장이 청취상 약해 정상본과 충분히
      구별되지 않음.

### 판정 원칙

- 연장의 절대 길이 또는 개인적으로 느끼는 길이감만으로 보류·제외하지 않는다.
- 목표 모음이 정상본과 인접 발화 단위보다 상대적으로 길고,
  청취상 자연스럽고 연속적으로 이어지면 통과한다.
- 음높이가 완만히 오르내리는 현상만으로는 끊김·합성 오류로 보지 않는다.
- 반복, 재시작, 분절 또는 발음 붕괴가 청취상 분명할 때 제외한다.

### 후속 확인

- 제외 파일은 필요 시 정상본 대비 파형·스펙트로그램으로 단절 또는
  연장 인지 약화 양상을 보조 기록할 수 있다.
- 시각 자료는 청취 판정의 보조 근거로만 사용하며, 청취상 분리된 파일을
  그림만으로 통과 처리하지 않는다.
- `spkS009`, `spkS010`에 같은 청취 기준을 적용한다.


### spkS009 청취 QC

### 대상
- 화자: `spkS009`
- 대상 파일: 11개
- 목적: `spkS001`에서 선택한 연장 표기가 다른 화자에서도
  자연스럽고 연속적인 연장으로 합성되는지 청취로 확인한다.
- 판정: `통과 / 보류 / 제외`

### 결과

- 통과: 9개
  - `spkS009__sent_01__target_01__C.wav`
  - `spkS009__sent_02__target_01__B.wav`
  - `spkS009__sent_02__target_02__C.wav`
  - `spkS009__sent_03__target_01__C.wav`
  - `spkS009__sent_03__target_02__B.wav`
  - `spkS009__sent_04__target_01__C.wav`
  - `spkS009__sent_05__target_01__C.wav`
  - `spkS009__sent_05__target_02__B.wav`
  - `spkS009__sent_06__target_01__C.wav`

- 제외: 2개
  - `spkS009__sent_04__target_02__C.wav`
    - 사유: “서.어.로의”처럼 목표 구간이 여러 덩이로 끊겨 들림.
  - `spkS009__sent_06__target_02__C.wav`
    - 사유: “조-오.오은”처럼 목표 구간이 끊겨 들림.


### spkS010 청취 QC

### 대상
- 화자: `spkS010`
- 대상 파일: 11개
- 목적: `spkS001`에서 선택한 연장 표기가 다른 화자에서도
  자연스럽고 연속적인 연장으로 합성되는지 청취로 확인한다.
- 판정: `통과 / 보류 / 제외`

### 결과

- 통과: 8개
  - `spkS010__sent_01__target_01__C.wav`
  - `spkS010__sent_02__target_01__B.wav`
  - `spkS010__sent_02__target_02__B.wav`
  - `spkS010__sent_03__target_01__C.wav`
  - `spkS010__sent_04__target_02__B.wav`
    - 청취에서는 “서.어어.로의”처럼 들려 연속성 여부가 애매했음.
    - 정상본과의 파형·스펙트로그램 비교에서 목표 구간의 유성 에너지와
      조화파가 완전히 끊겼다가 재시작하는 뚜렷한 양상은 확인되지 않음.
    - 음높이·에너지 변화가 있는 연속 연장으로 판단해 통과함.
  - `spkS010__sent_05__target_01__C.wav`
  - `spkS010__sent_05__target_02__C.wav`
  - `spkS010__sent_06__target_01__C.wav`

- 제외: 3개
  - `spkS010__sent_03__target_02__C.wav`
    - 사유: “오~~오, 오늘”처럼 목표 위치가 끊겨 들림.
  - `spkS010__sent_04__target_01__C.wav`
    - 사유: “말 하아~아~아~~~아~~~~”처럼 여러 번 분절·반복되어
      합성 오류로 판단됨.
  - `spkS010__sent_06__target_02__B.wav`
    - 사유: “조.오.오.흔”처럼 목표 위치가 분리되어 들림.


### 다화자 파일럿 QC 집계

- 대상: `spkS006`, `spkS009`, `spkS010`의 11개 파일씩, 총 33개
- 최종 결과: 통과 25개, 제외 8개, 보류 0개
- 화자별 결과:
  - `spkS006`: 통과 8개, 제외 3개
  - `spkS009`: 통과 9개, 제외 2개
  - `spkS010`: 통과 8개, 제외 3개
- `spkS010__sent_04__target_02__B.wav`는 청취상 연속성 여부가
  애매했으나, 정상본 대비 파형·스펙트로그램에서 완전한 유성 단절이나
  재시작이 뚜렷하지 않아 연속 연장으로 판단해 통과했다.
- 비교 그림:
  `data/prolong/text_pilot/qc_plots/spkS010__sent_04__normal_vs_target_02_B.png`

| 문장·위치 | 통과 / 전체 | 판단 |
|---|---:|---|
| `sent_01 target_01` | 3 / 3 | 유지 후보 |
| `sent_02 target_01` | 3 / 3 | 유지 후보 |
| `sent_02 target_02` | 3 / 3 | 유지 후보 |
| `sent_03 target_01` | 3 / 3 | 유지 후보 |
| `sent_03 target_02` | 1 / 3 | 재시험 후보 |
| `sent_04 target_01` | 2 / 3 | 재시험 후보 |
| `sent_04 target_02` | 1 / 3 | 재시험 후보 |
| `sent_05 target_01` | 3 / 3 | 유지 후보 |
| `sent_05 target_02` | 3 / 3 | 유지 후보 |
| `sent_06 target_01` | 3 / 3 | 유지 후보 |
| `sent_06 target_02` | 0 / 3 | 제외 |

### 다화자 파일럿 기반 생성 결정

- 현재 표기로 유지 후보:
  - `sent_01 target_01`
  - `sent_02 target_01`
  - `sent_02 target_02`
  - `sent_03 target_01`
  - `sent_05 target_01`
  - `sent_05 target_02`
  - `sent_06 target_01`

- 현재 표기 재시험 후보:
  - `sent_03 target_02`
  - `sent_04 target_01`
  - `sent_04 target_02`

- 현재 표기 제외:
  - `sent_06 target_02`

- 결정 근거:
  - 유지 후보는 3명 파일럿에서 모두 청취상 자연스럽고 연속적인
    연장으로 통과했다.
  - 재시험 후보는 한 명 이상에서 분절·반복 또는 합성 오류가 발생해
    현재 표기를 71명에게 일괄 적용하지 않는다.
  - `sent_06 target_02`는 세 화자 모두에서 연장 인지가 약하거나
    분절되어 현재 표기를 사용하지 않는다.
  - 이 결과는 3명 다화자 파일럿에 근거한 잠정 생성 규칙이다.
    71명 전체에서의 자동 통과를 의미하지 않는다.



## 2026-10-03 — prolongation v1 대량생성 규칙 재결정

### 확인한 입력·출력 경로

- 승인 화자 목록:
  - `data/_speaker_screening/approved_speakers_v1.csv`
- 다화자 파일럿 manifest:
  - `data/prolong/` 아래의 `multispeaker_pilot_manifest.csv`
- 다화자 파일럿 QC:
  - `metadata/prolong_multispeaker_pilot_qc.csv`
- 정식 v1 출력 루트:
  - `G:\내 드라이브\tts_dataset\synthetic_v1\`
- 계획 WAV 출력:
  - `audio/prolongation/`
- 계획 metadata 출력:
  - `metadata/prolongation_v1_manifest.csv`
  - `metadata/prolongation_v1_generation_log.csv`

### 파일럿 QC 해석

- 다화자 청취 QC를 문장 내 **연장 위치 기준**으로 집계하면 유지 후보는 7개다.
  - `sent_01 target_01`
  - `sent_02 target_01`
  - `sent_02 target_02`
  - `sent_03 target_01`
  - `sent_05 target_01`
  - `sent_05 target_02`
  - `sent_06 target_01`
- 그러나 `(sentence_id, target_id, variant)`까지 동일한 **고정 표기 기준**으로 보면, 아래 5개 규칙만 `spkS006`, `spkS009`, `spkS010` 모두에서 통과했다.
  - `sent_01 target_01 C`
  - `sent_02 target_01 B`
  - `sent_03 target_01 C`
  - `sent_05 target_01 C`
  - `sent_06 target_01 C`

### 7개와 5개의 차이

- `sent_02 target_02`는 위치 자체는 세 화자 모두 통과했지만, 통과한 variant가 화자별로 달랐다.
  - `spkS006`: `B` 통과
  - `spkS009`: `C` 통과
  - `spkS010`: `B` 통과
- `sent_05 target_02`도 위치 자체는 세 화자 모두 통과했지만, 통과한 variant가 화자별로 달랐다.
  - `spkS006`: `C` 통과
  - `spkS009`: `B` 통과
  - `spkS010`: `C` 통과
- 따라서 두 위치는 연장이 자연스럽게 합성될 수 있는 위치라는 근거는 있으나, 하나의 동일 B/C 표기를 모든 화자에 적용해도 안정적이라는 근거는 현재 부족하다.

### v1 생성 결정

- prolongation v1은 합성 품질을 우선하는 보수적 기준선으로 구성한다.
- 자동 QC 집계로 규칙을 선택하지 않고, 아래 5개 규칙을 생성 코드에 명시적으로 고정한다.
  - `sent_01 target_01 C`
  - `sent_02 target_01 B`
  - `sent_03 target_01 C`
  - `sent_05 target_01 C`
  - `sent_06 target_01 C`
- 대량생성 목표:
  - 승인 화자 71명 × 고정 규칙 5개 = **355개**
- v1 규칙은 동일 `(sentence_id, target_id, variant)`가 3명 파일럿에서 모두 청취 통과한 경우만 포함한다.
- 이 “3명 모두 통과” 기준은 학술적·임상적 연장 임계값이 아니라, 현재 파일럿의 합성 품질관리 목적을 위한 프로젝트 내부의 보수적 운영 기준이다.

### 보류 규칙과 향후 확장

- 아래 두 위치는 v1 생성에서는 보류하고, 후속 파일럿 및 확장 데이터 후보로 남긴다.
  - `sent_02 target_02`
  - `sent_05 target_02`
- 향후 더 다양한 화자에서 각 위치의 B/C 표기를 다시 생성·청취하고, 동일 표기가 여러 화자에서 자연스럽고 연속적으로 통과하는지 확인한다.
- 통과한 규칙은 v1.1 또는 v2의 위치 다양화 데이터로 추가 검토한다.
- `sent_03 target_02`, `sent_04 target_01`, `sent_04 target_02`는 재시험 후보 상태를 유지한다.
- `sent_06 target_02`는 현재 표기에서 제외 상태를 유지한다.


### 코드 수정 및 dry-run 검증

- `src/v1/generate_prolongation_v1.py`에서 QC CSV의 자동 집계로 생성 규칙을 선택하던 로직을 제거했다.
- 다화자 파일럿에서 동일 `(sentence_id, target_id, variant)`가 3명 모두 청취 통과한 5개 규칙을 코드 상수로 명시했다.
- pilot manifest는 규칙 선택용이 아니라, 확정 규칙의 `variant_text`, `normal_text`, `target_original`을 읽어 실제 Typecast 입력 텍스트를 재현하는 용도로 유지했다.
- `--dry-run` 결과:
  - 승인 화자 수: 71명
  - 고정 v1 규칙 수: 5개
  - 전체 생성 계획 수: 355개
  - Typecast API 호출: 0회
  - WAV 저장: 0개
  - manifest 및 generation log 저장: 0개
- dry-run에서 확인한 출력 경로:
  - WAV: `G:\내 드라이브\tts_dataset\synthetic_v1\audio\prolongation\`
  - manifest: `G:\내 드라이브\tts_dataset\synthetic_v1\metadata\prolongation_v1_manifest.csv`
  - generation log: `G:\내 드라이브\tts_dataset\synthetic_v1\metadata\prolongation_v1_generation_log.csv`

### 대량생성 결과

- `--limit 2`로 소량 생성한 파일을 먼저 청취했다.
  - `spkS001__sent_01__target_01__C.wav`
  - `spkS001__sent_02__target_01__B.wav`
- 두 소량 생성 파일은 청취상 자연스럽고 연속적인 연장으로 확인했다.
- 이후 전체 대량생성을 실행했다.
- 처리 결과:
  - 전체 처리 수: 355개
  - 신규 생성: 353개
  - 기존 파일 skip: 2개
  - 실패: 0개
- 기존 skip 2개는 소량 생성 단계에서 이미 생성한 파일이며, 전체 생성 시 정상 WAV로 확인되어 재생성하지 않았다.
- 최종 연장 WAV 수는 355개다.

### 자동 기술 QC

- 자동 기술 QC 스크립트:
  - `src/v1/qc_prolongation_v1.py`
- QC는 WAV 원본을 수정·삭제하지 않고, 파일 수, 읽기 가능 여부, sample rate, channel 수, 길이, peak, 무음 비율을 검사했다.
- 초기 실행에서는 QC 코드의 예상 sample rate가 `24000 Hz`로 설정되어 있어, 실제 파일 형식과 달라 355개 전체가 REVIEW로 표시되었다.
- QC CSV에서 실제 sample rate를 집계한 결과, 355개 WAV 모두 `44100 Hz`임을 확인했다.
- 따라서 `EXPECTED_SAMPLE_RATE`를 `44100`으로 수정한 뒤 자동 QC를 다시 실행했다.
- 수정 후 결과:
  - 검사 대상: 355개
  - PASS: 332개
  - REVIEW: 23개
  - FAIL: 0개
- 23개 REVIEW의 사유는 `peak >= 0.999` 경고뿐이었다.
- 길이 분포:
  - 최소: 2.690초
  - 중앙값: 3.570초
  - 최대: 5.920초
- 기술 QC 결과:
  - `G:\내 드라이브\tts_dataset\synthetic_v1\qc\prolongation_v1_technical_qc.csv`
  - `G:\내 드라이브\tts_dataset\synthetic_v1\qc\prolongation_v1_technical_qc_summary.txt`

### clipping 정밀 검사

- peak가 높은 23개 REVIEW 파일을 대상으로, 최대 진폭 근처에서 파형이 여러 샘플 동안 평평하게 이어지는 실제 clipping 의심 구간이 있는지 정밀 검사했다.
- 정밀 검사 스크립트:
  - `src/v1/check_prolongation_clipping.py`
- 검사 기준:
  - `abs(sample) >= 0.999`인 near-full-scale 구간을 확인했다.
  - 1.0 ms 이상 연속된 평탄 구간이 있으면 REVIEW 대상으로 표시하도록 했다.
  - 위 기준은 실제 clipping 후보를 보수적으로 찾기 위한 프로젝트 내부 기술 QC 값이며, 연장 판정 기준은 아니다.
- 검사 결과:
  - 검사 대상: 23개
  - PASS: 23개
  - REVIEW: 0개
  - FAIL: 0개
- 따라서 1차 QC에서 peak 경고가 있던 23개도 실제 파형 clipping이 확인되지 않았다.
- clipping 정밀 검사 결과:
  - `G:\내 드라이브\tts_dataset\synthetic_v1\qc\prolongation_v1_clipping_check.csv`
  - `G:\내 드라이브\tts_dataset\synthetic_v1\qc\prolongation_v1_clipping_check_summary.txt`

### 청취 표본 QC

- 기술 QC 이후, 대량 생성 결과의 위치별 합성 품질을 탐색적으로 확인하기 위해 5개 고정 규칙마다 승인 화자 목록의 정렬상 첫 화자와 마지막 화자를 선택했다.
- 청취 표본:
  - 화자: `spkS001`, `spkS080`
  - 규칙: 5개
  - 총 표본 수: 10개
- 청취 표본 목록 생성 스크립트:
  - `src/v1/make_prolongation_v1_listening_sample.py`
- 청취 판정 기준:
  - `pass`: 목표 모음이 자연스럽고 연속적으로 길게 들림
  - `review`: 연속성·연장 강도·자연스러움이 애매함
  - `exclude`: 반복·재시작·명확한 분절·발음 붕괴 또는 연장 인지 약화
- 청취 결과:
  - PASS: 10개
  - REVIEW: 0개
  - EXCLUDE: 0개
- 모든 표본은 청취상 자연스럽고 연속적인 연장으로 기록했다.
- 청취 QC 기록:
  - `G:\내 드라이브\tts_dataset\synthetic_v1\qc\prolongation_v1_listening_sample.csv`
  - `G:\내 드라이브\tts_dataset\synthetic_v1\qc\prolongation_v1_listening_sample_instructions.txt`

### v1 최종 상태

- prolongation v1 대량생성은 완료되었다.
- 최종 확보한 연장 WAV는 355개다.
- 모든 파일은 읽기 가능한 WAV로 확인되었고, 기술 QC에서 FAIL은 0개였다.
- peak 경고 23개는 clipping 정밀 검사에서 모두 PASS로 확인되었다.
- 5개 규칙 × 2개 화자, 총 10개 청취 표본은 모두 자연스럽고 연속적인 연장으로 PASS 판정을 받았다.
- 이 결과는 합성 데이터의 기술적 완전성과 탐색적 청취 품질을 확인한 것이다.
- 10개 표본 청취는 355개 전체의 자연스러움을 전수 보증하는 검사는 아니므로, 이후 CNN 학습 전·중에 이상 사례가 확인되면 해당 화자·규칙 조합을 추가 청취한다.
- 연장 라벨은 합성 입력 텍스트에 기반한 프로젝트 내부 조작 라벨이며, 실제 개인의 불안도, 정신건강 상태, 면접 역량, 임상적 말더듬 또는 임상적 연장을 진단·판정하지 않는다.


## 2026-10-04 — slow-normal v1 생성 및 기술 QC 완료

### 목적

- 국소 연장(prolongation) 검출 모델이 문장 전체의 느린 발화를
  특정 모음·음절의 국소 연장으로 오탐하지 않도록,
  `prolongation_label=0` slow-normal 반례 음성을 생성한다.
- slow-normal은 실제 화자의 느린 발화, 불안도 또는 임상 상태를 나타내는
  관측 데이터가 아니라, normal_energy TTS 원본에 시간축 변환을 적용한
  합성 데이터 조건이다.

### 입력 및 변환

- 입력 원본: `synthetic_v1/audio/normal_energy/`
- 원본 수: 426개 WAV
- 출력: `synthetic_v1/audio/slow_normal/`
- 적용 함수: `librosa.effects.time_stretch`
- 적용 rate: `0.85`
- 이론상 길이 비율: \(1 / 0.85 = 1.17647\)
- 파일명 매핑:
  - 원본: `spkS001__sent_01__normal_energy.wav`
  - 출력: `spkS001__sent_01__slow_normal_r085.wav`
- 원본 normal_energy WAV는 수정·삭제·덮어쓰기하지 않았으며,
  slow-normal WAV를 별도 폴더에 생성했다.

### rate 선정 근거

- normal 원본 3개에 `rate=0.85`를 적용한 파일럿 청취를 수행했다.
- 문장 전체가 비교적 고르게 느려졌고, 특정 모음 또는 음절만 국소 연장처럼
  현저하게 두드러지는 현상은 청취상 확인되지 않았다.
- 정식 출력 경로에서 2개를 추가 생성해 재청취했으며, 동일하게
  전역 감속으로 판단했다.
- `rate=0.85`는 음성학적·임상적 느린 발화 절단점이 아니라,
  slow-normal 반례 효과를 확보하기 위해 파일럿 청취를 바탕으로 선택한
  프로젝트 내부의 잠정 합성 파라미터다.

### 생성 결과

| 항목 | 결과 |
|---|---:|
| 처리 대상 | 426개 |
| 신규 생성 | 424개 |
| 기존 파일 skip | 2개 |
| 생성 실패 | 0개 |
| 최종 slow-normal 파일 수 | 426개 |

### 기술 QC

- QC 스크립트: `src/v1/qc_slow_normal_v1.py`
- QC 상세 결과: `synthetic_v1/metadata/slow_normal_v1_qc_report.csv`
- QC 요약: `synthetic_v1/metadata/slow_normal_v1_qc_summary.txt`
- 기술 QC 결과: **PASS 426 / FAIL 0**

검사 항목:

- normal_energy 원본과 slow-normal 출력의 1:1 파일 대응
- WAV 파일 존재 및 `soundfile`·`wave` 라이브러리로 읽기 가능 여부
- 빈 오디오, NaN, Inf 여부
- 원본과 출력의 sample rate 및 채널 수 일치 여부
- slow-normal/normal_energy 길이 비율
- slow-normal RMS 및 무음 비율

길이 비율 기술 QC 기준:

- 허용 범위: `1.10–1.30`
- 이론값 근접 기준: `1.17647 ± 0.03`
- 위 기준은 `rate=0.85` 시간축 변환이 의도대로 적용되었는지를 확인하는
  프로젝트 내부 잠정 기술 기준이며, 임상적 말속도·느린 발화 판정 기준은 아니다.

## 2026-10-04 — 청취 QC 진행 현황 

### 대표 표본 청취 QC

- 기술 QC PASS 426개 중 화자·문장 위치가 고르게 포함되도록 선정한
  대표 표본 18개를 청취했다.
- 청취 항목은 재생 가능 여부, 전역 감속의 일관성, 특정 음절·모음의
  국소 연장처럼 들리는 구간 유무, 심한 금속성·울림·분절 왜곡 여부다.
- 대표 표본 18개 중 16개는 `PASS`, 2개는 `HOLD`, `FAIL`은 0개였다.
- 대표 표본 청취 결과는
  `synthetic_v1/metadata/slow_normal_v1_listening_qc_sample.csv`에 기록했다.

### HOLD 파일 및 전수 청취 전환 사유

- `spkS080__sent_01__slow_normal_r085.wav`
  - “안녕하세요.” 말미에서 원본에도 약한 길이 늘임이 들리지만,
    slow-normal 변환 후에는 국소 연장처럼 지각될 가능성이 있어 `HOLD`로 기록했다.
  - 문장 종결 위치의 자연스러운 운율적 길이 늘임인지, 모델 학습에서
    국소 연장과 혼동될 수준인지 원본 normal_energy와 재비교가 필요하다.

- `spkS080__sent_06__slow_normal_r085.wav`
  - “더”가 강조 발화처럼 상대적으로 길게 들려 국소 연장과 혼동될
    가능성이 있어 `HOLD`로 기록했다.

- 대표 표본에서 `HOLD`가 2개 발생했으므로, `rate=0.85` 변환 파일
  426개 전체를 청취 확인 없이 일괄적으로 `prolongation_label=0`으로
  확정하지 않기로 결정했다.
- 이에 따라 전체 slow-normal 파일을 순차 청취하고, 특이·보류·제외
  후보를 추가 확인하는 전수 청취 QC로 전환한다.
- 이 결정은 `rate=0.85` time-stretch 자체가 기술적으로 실패했다는
  뜻이 아니라, 원본의 강조·문장 경계 말미 길이 늘임이 전역 감속 후
  국소 연장과 혼동될 수 있는 사례를 보수적으로 제외하기 위한 것이다.

### 전수 청취 QC 기록 방식

- 전수 청취에서 특이사항 없이 적합한 파일은 별도 행을 추가하지 않는다.
- 애매하거나 학습 반례로 부적절할 수 있는 파일만
  `synthetic_v1/metadata/slow_normal_v1_listening_qc_review_log.csv`에
  `HOLD` 또는 `FAIL`로 기록한다.
- `HOLD`는 즉시 제외하지 않고 원본 normal_energy와 비교 재청취한 뒤
  최종 `PASS` 또는 `FAIL`로 확정한다.
- 청취 QC가 완료된 후에는 기술 QC PASS 및 최종 청취 PASS 파일만
  `prolongation_label=0` slow-normal 학습 후보 manifest에 포함한다.

### 후속 작업

- slow-normal 426개를 순차 청취하고, HOLD·FAIL 후보를 review log에 기록한다.
- HOLD 파일을 원본 normal_energy와 비교 재청취해 최종 PASS 또는 FAIL로 확정한다.
- 최종 청취 PASS 파일을 기준으로 slow-normal v1 학습 후보 manifest를 생성한다.
- 청취 QC 최종 결과와 포함·제외 수를 experiment log에 추가 기록한다.


## 2026-10-05 — tremor v1 파일럿: round2 문장 다양성 및 round3 강도 증강

### 목적

- 1차 파일럿에서 우선 후보로 선정된 `T04_clear_5p0`이 화자와 문장이 달라져도 자연스럽게 지각되는지 확인했다.
- 2차 청취에서 S001·S002·S004의 tremor 지각성이 약하게 관찰되어, 동일한 5.0 Hz rate에서 F0·amplitude 변조폭만 증가한 조건을 추가 생성했다.
- 본 기록의 T04·T05·T06 파라미터는 면접 음성 훈련용 합성 파일럿의 프로젝트 내부 탐색값이며, 임상 tremor·불안·질환의 진단 또는 중증도 기준이 아니다.

### round2: 문장·화자 일반화 확인

- 생성 스크립트: `src/v1/make_tremor_v1_pilot_round2.py`
- 기술 QC 스크립트: `src/v1/qc_tremor_v1_pilot_round2.py`
- 대상: `S001`~`S004` × `sent_01`·`sent_03`·`sent_06` × `T04_clear_5p0`
- 생성 수: 4화자 × 3문장 × 1조건 = 12개 WAV
- 출력 폴더: `data/_pilot_tremor_v1_round2/`
- 매니페스트: `data/_pilot_tremor_v1_round2/tremor_v1_round2_manifest.csv`

#### T04 합성 파라미터

- `rate_hz=5.0`
- `f0_depth_semitones=1.10`
- `amplitude_depth=0.14`
- `amplitude_phase_rad=0.0`
- `rate_jitter_hz=0.12`

#### round2 기술 QC 결과

- 기술 QC 결과: 12/12 `PASS`
- 검토 flag: 0개
- QC CSV: `data/_pilot_tremor_v1_round2/tremor_v1_round2_technical_qc.csv`
- QC 요약: `data/_pilot_tremor_v1_round2/tremor_v1_round2_technical_qc_summary.txt`
- 조건별 평균:
  - `mean_output_f0_sd_st=4.963993`
  - `mean_f0_band_ratio=0.226222`
  - `mean_amp_band_ratio=0.436415`
- 해석: 위 3–7 Hz peak/band power ratio는 5 Hz 중심의 F0·amplitude 변조가 신호에서 분석 가능한지 확인하는 내부 기술 QC 지표다. 문장 고유의 억양 및 유성구간 단절의 영향을 받으므로, 조건의 지각적 강도 또는 임상 상태를 직접 의미하지 않는다.

#### round2 청취 QC 결과

- S003: `sent_01`, `sent_03`, `sent_06`에서 tremor가 상대적으로 적절하게 지각되었다.
- S001, S002, S004: 세 문장에서 T04 tremor가 전반적으로 약하게 지각되었다.
- 뚜렷한 click, pumping, 금속성, 분절 왜곡 등 합성 artifact는 청취에서 보고되지 않았다.
- 결론: T04는 기술적으로 정상이나, 동일 고정 파라미터가 모든 화자에서 균일한 tremor 지각성을 제공하지 못했다. S003은 T04 유지 후보로, S001·S002·S004는 강도 증강 비교 대상으로 분류했다.

### round3: S001·S002·S004 강도 증강 미니 파일럿

- 생성 스크립트: `src/v1/make_tremor_v1_pilot_round3_boost.py`
- 기술 QC 스크립트: `src/v1/qc_tremor_v1_pilot_round3_boost.py`
- 대상: `S001`·`S002`·`S004` × `sent_01`
- 비교 조건: `T05_boost_5p0`, `T06_strong_5p0`
- 생성 수: 3화자 × 1문장 × 2조건 = 6개 WAV
- 출력 폴더: `data/_pilot_tremor_v1_round3_boost/`
- 매니페스트: `data/_pilot_tremor_v1_round3_boost/tremor_v1_round3_boost_manifest.csv`

#### round3 합성 파라미터

| condition_id | rate_hz | F0 depth | amplitude depth | amplitude phase |
|---|---:|---:|---:|---:|
| `T05_boost_5p0` | 5.0 | ±1.40 st | 0.18 | 0.0 rad |
| `T06_strong_5p0` | 5.0 | ±1.70 st | 0.22 | 0.0 rad |

- 두 조건 모두 `rate_jitter_hz=0.12`를 사용했다.
- T05/T06 수치는 T04보다 변조폭을 점진적으로 증가시킨 프로젝트 내부 청취 비교용 탐색값이다.
- rate를 5.0 Hz로 고정해, 청취 차이가 발생할 경우 rate보다 변조폭 변화의 영향을 우선 비교할 수 있도록 설계했다.

#### round3 기술 QC 결과

- 기술 QC 결과: 6/6 `PASS`
- 검토 flag: 0개
- QC CSV: `data/_pilot_tremor_v1_round3_boost/tremor_v1_round3_boost_technical_qc.csv`
- QC 요약: `data/_pilot_tremor_v1_round3_boost/tremor_v1_round3_boost_technical_qc_summary.txt`
- 현재 상태: 기술 QC 완료, T04·T05·T06 비교 청취 QC는 미완료.

### 다음 작업

- S001·S002·S004 각각에서 동일 문장 `sent_01`의 T04 → T05 → T06을 비교 청취한다.
- 각 조건에 대해 tremor 지각성(약함/적절/과도), artifact 여부, 화자별 선택 조건을 기록한다.
- 청취 결과를 바탕으로 S001·S002·S004에 적용할 보정 조건을 결정하거나, 필요 시 추가 강도 파일럿을 설계한다.
- S003은 T04를 유지할 수 있는지 최종 프로젝트 정책과 데이터 균형을 고려해 결정한다.

### 근거 및 해석 한계

- Vocal tremor는 F0와 loudness/intensity의 준주기적 변조로 기술된다.
- simulated tremor 연구에서 3·5·7 Hz 변조 조건이 사용되었으며, 본 프로젝트는 5 Hz를 합성 탐색 중심값으로 유지했다.
- 문헌은 tremor 지각에 F0 및 intensity modulation extent가 중요할 수 있음을 제시하지만, 본 프로젝트의 F0 depth와 amplitude depth는 환자 음성을 그대로 재현한 임상 표준값이 아니다.
- 본 파일럿의 3–7 Hz band ratio와 청취 결과는 합성 학습 데이터의 품질·지각성 확인을 위한 것으로, 개인의 불안, 질환, 또는 임상 tremor 여부·중증도를 판정하지 않는다.

## 2026-10-06 — round3 청취 QC 결과
---

## 2026-10-07 — Tremor v1 전체 기술 QC 1차 결과

### 실행 스크립트
- `src/v1/qc_tremor_v1.py`

### 검사 대상 및 산출물
- 검사 대상: `tremor_v1` WAV 420개
- 입력 manifest: `synthetic_v1/metadata/tremor_v1_manifest.csv`
- 파일별 QC 결과: `synthetic_v1/metadata/tremor_v1_qc.csv`
- flag 목록: `synthetic_v1/metadata/tremor_v1_qc_flags.csv`
- 요약: `synthetic_v1/metadata/tremor_v1_qc_summary.txt`

### 1차 파일·파형 무결성 결과
- 검사 수: 420개
- hard fail: 0개
- 원본-출력 길이 차이: 평균/중앙값/최소/최대 모두 0.0 ms
- RMS 비율: 평균 0.9599, 중앙값 1.0000, 최소 0.6641, 최대 1.0000
- 출력 peak: 평균 0.8802, 중앙값 0.9102, 최소 0.5373, 최대 0.9800
- voiced ratio: 평균 0.7499, 중앙값 0.7587, 최소 0.5044, 최대 0.8993
- peak > 0.99: 0개
- RMS ratio < 0.70: 2개. 자동 제외하지 않고 원본-출력 쌍 청취 QC 대상으로 보류함

### 변조율 분석 결과와 해석
- F0 modulation off-target: 391개
- amplitude modulation off-target: 355개
- 기존 QC의 soft flag: 418개, flag 없음: 2개
- F0 modulation rate: 평균 7.2386 Hz, 중앙값 10.0000 Hz, 범위 2.1505–10.0000 Hz
- amplitude modulation rate: 평균 3.8806 Hz, 중앙값 2.6316 Hz, 범위 2.0000–10.0000 Hz

현재 QC는 문장 전체의 F0 contour와 RMS envelope에 자기상관 기반 단일 지배 변조율을 적용했다. 그러나 연결 발화에는 무성구간, 휴지, 음절 리듬, 억양 변화 및 F0 추적 오차가 포함되어, 주입한 5 Hz tremor가 단일 지배 피크로 검출되지 않을 수 있다. F0 추정 중앙값이 탐색 상한인 10 Hz에 집중된 현상도 상한 포화형 추정 artifact로 해석한다.

따라서 `f0_modulation_off_target` 및 `amplitude_modulation_off_target`는 현재 단계에서 생성 실패 또는 자동 제외 근거로 사용하지 않는다. 파일 무결성·시간 정합성·sample rate·peak 기준은 420개 전체 통과로 판정한다.

### 근거
- 연결 발화보다 지속 모음 또는 안정된 유성구간에서 F0 및 intensity tremor modulation rate/extent를 분석하는 방식이 널리 사용된다.
- Sustained vowel 기반 연구에서 F0 및 intensity modulation rate는 각각 약 4.6 Hz, 5.2 Hz로 보고되었다.
- 출처: [Perceptual and Acoustical Features of Dysarthria in Essential Tremor, 2026](https://tremorjournal.org/articles/10.5334/tohm.1180)
- 출처: [Physiologic and Acoustic Patterns of Essential Vocal Tremor, Lester-Smith et al., 2013](https://experts.arizona.edu/en/publications/physiologic-and-acoustic-patterns-of-essential-vocal-tremor/)

### 후속 조치
1. RMS ratio < 0.70인 2개 파일을 원본-출력 쌍으로 청취한다.
2. 연결 발화 전체의 단일 modulation-rate 기반 soft flag는 최종 합격/불합격 기준에서 제거한다.
3. 이후 분석 지표는 연속 유성구간 기반 5 Hz 대역 power, F0/amplitude modulation extent, 청취 QC와 함께 해석한다.
4. 1차 QC CSV와 summary는 재현 기록으로 보존한다.

- 대상: S001·S002·S004 × `sent_01` × T04·T05·T06 비교
- T04는 세 화자에서 공통적으로 tremor 지각성이 약했다.
- T05와 T06은 세 화자에서 모두 적절한 tremor로 지각되었다.
- T06은 T05보다 tremor가 더 명확하게 지각됐으나, 청취상 과도한 비브라토·기계적 진동·전달력 저하는 관찰되지 않았다.
- T05와 T06의 뚜렷한 click, pumping, 금속성, 분절 왜곡은 관찰되지 않았다.
- 결론: S001·S002·S004에는 T04 대신 T05와 T06을 파일 단위로 병행 적용하는 방안을 대량 생성 전 후보 정책으로 검토한다.
- T05:T06의 초기 배분은 1:1을 잠정 제안하되, 이는 문헌 기반 임상 분포가 아니라 강도 다양성 확보를 위한 프로젝트 내부 균형값이다.
- round3 기술 QC는 T05·T06 총 6개 파일에서 6/6 PASS였고, 검토 flag는 0개였다.
- 현재 청취 결과는 S001·S002·S004 × `sent_01`에서만 확인된 결과이므로, 다른 문장 및 파일럿에 포함되지 않은 화자에 동일 정책을 일반화하기 전 추가 검증이 필요하다.
- 대량 생성 시에는 출력 WAV별로 `speaker_id`, `sentence_id`, `condition_id`, `rate_hz`, `f0_depth_semitones`, `amplitude_depth`, 청취 QC 결과를 manifest에 기록한다.


### round4 대표 화자 청취 QC 결과

- 대상: S005·S011·S017·S024·S031·S038·S044·S050·S057·S065·S072·S080 × `sent_01` × T05·T06 비교
- 생성 수: 12화자 × 1문장 × 2조건 = 24개 WAV
- 기술 QC: 24/24 PASS, 검토 flag 0개
- S005, S011, S017, S024, S031, S038, S044, S050, S057, S065, S072에서 T06의 tremor가 명확하게 지각되었고, 과도한 비브라토·기계적 진동·전달력 저하는 관찰되지 않았다.
- S080은 강세가 강한 구간에서 T05·T06 모두 소리가 부자연스럽게 잡히는 현상이 관찰되어, 두 조건 모두 HOLD로 기록한다.
- S080의 현상은 자동 기술 QC에서 flag로 검출되지 않은 국소적 청취 artifact 가능성이므로, 대량 생성 후보에서 우선 제외하고 원본 normal_energy와 A/B 비교 재청취한다.
- 결론: T06_strong_5p0을 전체 tremor 대량 생성의 기본 후보로 선정한다. 단, S080은 별도 재검토가 완료될 때까지 대량 생성 대상에서 제외한다.
- T05는 현재 대량 생성 기본 조건으로 사용하지 않고, 향후 tremor 강도 다양성 레벨을 추가할 때 보조 후보로 보존한다.
- 본 결정은 파일럿 청취 및 내부 기술 QC에 근거한 프로젝트 내부 합성 정책이며, 임상 tremor·불안·질환의 진단 또는 중증도 기준이 아니다.
- S080은 원본 normal_energy에도 강세가 강한 구간이 존재했다.
- T05·T06 적용 후 해당 구간에서 강세·에너지 변화와 tremor 변조가 겹치며,
  청취상 매우 부자연스러운 국소 artifact가 관찰되었다.
- 이 현상은 자동 기술 QC에서는 flag로 검출되지 않았으나, 학습 데이터에
  포함할 경우 비의도적 음향 패턴을 학습시킬 위험이 있어 S080은
  tremor 대량 생성 대상에서 제외한다.
- S080 원본 normal_energy 파일은 변경·삭제하지 않으며, tremor 파생본만
  제외한다.

### 2026-10-06 — Tremor v1 대량 생성 완료

### 목적
`normal_energy` 원본 음성에 tremor 합성 조건 `T06_strong_5p0`을 일괄 적용하여
tremor v1 학습/평가용 파생 음성을 생성한다.

### 실행 스크립트
- `src/v1/generate_tremor_v1.py`

### 입력
- 입력 폴더: `G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy\`
- 입력 파일 패턴: `*__normal_energy.wav`
- 입력 원본 수: 426개
- 구성: 71명 × 6문장

### 적용 조건
- condition ID: `T06_strong_5p0`
- tremor rate: 5.0 Hz
- rate jitter: 0.12 Hz
- F0 modulation depth: ±1.70 semitone
- amplitude modulation depth: 0.22
- amplitude phase: 0.0 rad
- WORLD frame period: 5.0 ms
- F0 search range: 60–500 Hz
- random seed: `20261006`

> 위 파라미터는 round3/round4의 프로젝트 내부 파일럿 및 청취 QC를 바탕으로 선택한 합성 조건이다. 임상 vocal tremor의 진단 또는 중증도 판정 기준은 아니다.

### 제외 처리
- 제외 화자: `S080`
- 제외 파일 수: 6개
- 제외 사유: 원본의 강한 강세 구간과 tremor 변조가 결합할 때 국소 청취 artifact가 관찰됨
- 원본 `normal_energy` 파일은 수정·삭제하지 않았음
- 제외 상세: `synthetic_v1/metadata/tremor_v1_excluded_sources.csv`

### 생성 결과
- 생성 완료 WAV: 420개
- 최종 구성: 70명 × 6문장
- 출력 폴더: `G:\내 드라이브\tts_dataset\synthetic_v1\audio\tremor_v1\`
- manifest: `G:\내 드라이브\tts_dataset\synthetic_v1\metadata\tremor_v1_manifest.csv`
- 출력 형식: mono, PCM_16 WAV
- 출력 파일명 형식:
  `spkS001__sent_01__normal_energy__tremor_T06_strong_5p0.wav`

### 생성 안전장치 및 결과
- 생성 전 원본 426개, S080 제외 6개, 생성 대상 420개를 검증함
- 기존 결과 파일 및 기존 CSV가 존재하면 덮어쓰지 않고 중단하도록 구현함
- 출력 WAV마다 저장 후 읽기 가능 여부와 유한값 여부를 검사함
- 실행 결과:
  - `생성 완료: 420개 WAV`
  - `제외 기록: 6개`
- 생성 단계 오류 없이 완료됨

### 다음 작업
1. `src/v1/qc_tremor_v1.py`를 작성한다.
2. manifest 기준으로 tremor 출력 420개만 전체 기술 QC한다.
3. 길이, peak, RMS, 파일 손상 여부 및 변조 지표를 확인한다.
4. QC flag 파일을 만들어 청취 검토 우선순위를 정한다.
5. 기술 QC 통과 후 표본 청취 QC를 진행한다.


---

## 2026-10-07 — Tremor v1 전체 QC 및 수동 청취 검토

### QC 목적
`normal_energy` 원본을 기반으로 생성한 `tremor_v1` 420개에 대해 파일 무결성,
원본-출력 정합성, 레벨, 변조 분석 및 flag 파일 청취 검토를 수행했다.

### QC 스크립트 및 산출물
- 실행 스크립트: `src/v1/qc_tremor_v1.py`
- 검사 대상: `tremor_v1` WAV 420개
- 입력 manifest: `synthetic_v1/metadata/tremor_v1_manifest.csv`
- 파일별 QC 결과: `synthetic_v1/metadata/tremor_v1_qc.csv`
- flag 목록: `synthetic_v1/metadata/tremor_v1_qc_flags.csv`
- 요약: `synthetic_v1/metadata/tremor_v1_qc_summary.txt`

### 1차 기술 QC 결과
| 항목 | 결과 | 판정 |
|---|---:|---|
| 검사 파일 수 | 420개 | 완료 |
| hard fail | 0개 | 통과 |
| soft flag | 418개 | 변조율 자동 판정 과검출 포함 |
| flag 없음 | 2개 | - |
| 원본-출력 길이 차이 | 평균/중앙값/최소/최대 모두 0.0 ms | 통과 |
| RMS 비율 | 평균 0.9599, 중앙값 1.0000, 범위 0.6641–1.0000 | 2개 청취 검토 |
| 출력 peak | 평균 0.8802, 중앙값 0.9102, 범위 0.5373–0.9800 | 통과 |
| voiced ratio | 평균 0.7499, 중앙값 0.7587, 범위 0.5044–0.8993 | 참고 지표 |

### 변조율 flag 해석
- F0 modulation off-target: 391개
- amplitude modulation off-target: 355개
- F0 modulation rate: 평균 7.2386 Hz, 중앙값 10.0000 Hz, 범위 2.1505–10.0000 Hz
- amplitude modulation rate: 평균 3.8806 Hz, 중앙값 2.6316 Hz, 범위 2.0000–10.0000 Hz

기존 QC는 문장 전체 F0 contour와 RMS envelope에서 자기상관 기반 단일 지배 변조율을 추정하여,
5 Hz 합성 조건과의 차이를 soft flag로 표시했다. 그러나 연결 발화에는 무성구간, 휴지,
음절 리듬, 억양 변화 및 F0 추적 오차가 포함되므로 주입된 5 Hz 변조가 문장 전체에서
하나의 지배 피크로 검출되지 않을 수 있다. F0 modulation rate의 중앙값이 탐색 상한인
10 Hz에 집중된 현상은 상한 포화형 추정 artifact로 해석한다.

따라서 `f0_modulation_off_target` 및 `amplitude_modulation_off_target`는 생성 실패,
자동 제외 또는 재생성의 근거로 사용하지 않는다. 이 값들은 연결 발화 기반 탐색 지표로만
보존하며, 최종 판단에는 파일 무결성, 레벨 QC 및 청취 결과를 함께 사용한다.

> **판정 기준 주의:** length 10 ms, peak 0.99, RMS ratio 0.70–1.30은 임상 vocal tremor 진단 기준이 아니라 본 프로젝트의 잠정 기술 QC 기준이다. 연결 발화에서 F0·intensity tremor rate를 단일 수치로 자동 판정하는 방식은 제한적이므로, flag는 청취 검토 우선순위로만 사용했다.

### RMS 저하 후보 청취 QC

RMS ratio < 0.70으로 flag된 파일은 2개였다. 두 파일 모두 원본과 tremor 출력의
청취 비교를 수행했다.

| 대상 | 원본 상태 | tremor 출력 청취 결과 | 결정 | 후속 조치 |
|---|---|---|---|---|
| `S028 / sent_01` | 원본 길이 약 8초. 약 4초 이후 목표 한국어 문장과 무관한 타 언어 추가 녹음 확인 | tremor 출력에도 원본 후반부 추가 음성이 포함될 가능성 있음 | 보류 | 원본 절단 가능 여부 검토 후 해당 원본 기반 파생본을 별도 버전으로 재생성 |
| `S039 / sent_05` | normal_energy 원본은 내용·음질상 이상 없음 | “차분한 마음” 중 “차분한”의 강세 구간에서 국소적 음성 품질 이상 청취 | tremor 출력 제외 | tremor 조건 또는 국소 gain/F0 변조 방식을 조정한 후속 버전에서 재생성 여부 검토 |

### S028/sent_01 영향 범위

`S028 / sent_01`의 원본 내용 이상은 tremor QC 청취 과정에서 확인했다.
현재 원본과 기존 파생본을 삭제·수정하지 않으며, 후반부 비목표 추가 음성 구간 절단 및
파생본 재생성 여부는 별도 작업으로 보류한다.

| 계열 | 확인된 파일 |
|---|---|
| normal_energy 원본 | `audio/normal_energy/spkS028__sent_01__normal_energy.wav` |
| energy_fade_in | `audio/energy_fade_in/spkS028__sent_01__energy_fade_in.wav` |
| energy_fade_out | `audio/energy_fade_out/spkS028__sent_01__energy_fade_out.wav` |
| prolongation | `audio/prolongation/spkS028__sent_01__target_01__C.wav` |
| slow_normal | `audio/slow_normal/spkS028__sent_01__slow_normal_r085.wav` |
| tremor_v1 | `audio/tremor_v1/spkS028__sent_01__normal_energy__tremor_T06_strong_5p0.wav` |

- S028의 `sent_02`~`sent_06` normal_energy 원본은 청취 확인 결과 정상으로 판단했다.
- S028/sent_01 이슈 유형: `extra_non_target_speech_after_target_utterance`
- 현재 상태: `source_content_review_pending_trim_and_regeneration`

### 현 시점 데이터 상태

| 구분 | 수량 | 상태 |
|---|---:|---|
| tremor v1 생성 완료 | 420개 | 생성 기록 보존 |
| 파일 무결성 hard fail | 0개 | 전체 통과 |
| 변조율 기반 soft flag | 418개 | 연결 발화 자동 추정 한계로 자동 제외 근거에서 제거 |
| 원본 내용 이슈 보류 | 1개 | `S028 / sent_01` |
| tremor 합성 artifact 제외 | 1개 | `S039 / sent_05` |
| 현시점 즉시 사용 가능 후보 | 418개 | S028 보류 및 S039 제외를 반영한 잠정 수량 |

### 참고 근거
- Carbonell et al. (2015), *Discriminating Simulated Vocal Tremor Source Using Amplitude Envelope Spectral Measures*, *Journal of Voice*. F0 및 amplitude 변조를 포함한 simulated vocal tremor의 음향 분석을 다룸.  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4361255/
- Barkmeier-Kraemer & Clark (2010), *Conceptual and Clinical Updates on Vocal Tremor*, *The ASHA Leader*. Vocal tremor의 F0 및 intensity modulation rate에 대한 임상·음향적 논의.  
  https://leader.pubs.asha.org/doi/10.1044/leader.FTR2.15142010.16
- Lester-Smith et al. (2013), *Physiologic and Acoustic Patterns of Essential Vocal Tremor*, *Journal of Voice*. 지속 모음 기반으로 vocal tremor의 생리·음향 패턴을 분석함.  
  https://experts.arizona.edu/en/publications/physiologic-and-acoustic-patterns-of-essential-vocal-tremor/

### 다음 작업
1. `S028 / sent_01`의 목표 문장이 약 4초 이전에 완전히 끝나는지 확인하고, 절단 시점 및 원본 교정 방식을 결정한다.
2. 원본 교정이 확정되면 S028/sent_01 기반 파생본을 새 버전으로 재생성한다.
3. `S039 / sent_05` tremor 출력은 현 버전에서 제외하고, 국소 강세 구간 artifact 완화 로직을 검토한다.
4. 이후 QC 스크립트 개정 시 modulation-rate 값은 참고용 분석 열로 보존하되, 단일 off-target flag를 자동 제외 기준으로 사용하지 않는다.

---

### QC v2 실행 결과

기존 QC v1에서 연결 발화 전체의 단일 modulation-rate를 soft flag로 사용하면서
418개가 flag된 문제를 보완하기 위해 `qc_tremor_v1_v2.py`를 실행했다.

- 실행 스크립트: `src/v1/qc_tremor_v1_v2.py`
- 검사 대상: 420개
- 파일별 결과: `synthetic_v1/metadata/tremor_v1_qc_v2.csv`
- flag 목록: `synthetic_v1/metadata/tremor_v1_qc_v2_flags.csv`
- 요약: `synthetic_v1/metadata/tremor_v1_qc_v2_summary.txt`
- hard fail: 0개
- soft flag: 2개
- flag 없음: 418개

QC v2에서는 `f0_modulation_off_target` 및
`amplitude_modulation_off_target`를 자동 soft flag 조건에서 제거했다.
F0/amplitude modulation rate와 voiced ratio는 참고 분석 열로만 보존했다.

soft flag 2개는 RMS ratio < 0.70 후보이며, 이미 수동 청취 검토를 완료했다.
- `S028 / sent_01`: 원본 후반부 비목표 추가 음성 이슈로 trim 및 재생성 검토 보류
- `S039 / sent_05`: tremor 출력의 국소 강세 구간 artifact로 현 v1 출력 제외

따라서 현재 tremor v1의 즉시 사용 가능 후보는 418개다.

---

### 수동 검토 결정 파일
- 수동 청취 및 원본 내용 검토 결과는 `synthetic_v1/metadata/tremor_v1_manual_review.csv`에 기록했다.
- `decision=hold` 및 `decision=exclude` 파일은 현시점 tremor v1의 즉시 사용 대상에서 제외한다.

---

## 2026-10-10 — normal_energy 원본 426개 전수 청취 QC 완료

### 검토 범위
- 대상: `normal_energy` 원본 71명 × 6문장 = 총 426개 WAV
- 검토 방식: 모든 파일을 직접 청취하여 목표 문장 보존 여부,
  비목표 음성 포함 여부, 선행 무음 및 발화 특성을 확인했다.
- 기존의 “전체 원본 내용 청취 검증 미완료” 상태를 본 기록으로 갱신한다.

### 전수 청취 결과
- 내용·음질 측면의 합성 오류는 기존에 확인한 `S028 / sent_01`에서만 발견됐다.
- 나머지 425개에서는 이번 청취 검토상 별도의 합성 오류가 발견되지 않았다.
- 일부 정상 원본에서는 느린 발화, 말미 늘어짐, 억양 또는 강조 때문에
  특정 구간이 연장처럼 느껴지는 지점이 있었다.
- 이러한 지점은 이번 청취 검토에서 오류로 분류하지 않고 정상 데이터를 유지했다.

### 개별 확인 사항

| 대상 | 청취 관찰 | 현재 결정 | 후속 작업 |
|---|---|---|---|
| `S004 / sent_02` | 시작 부분에 약 1초의 선행 무음이 있으며, 이후 목표 문장은 누락 없이 발화됨. 총길이는 약 5초로 관찰됨 | 정상 데이터 유지 | 학습 입력 전처리에서 선행 무음 처리 여부 검토 |
| `S028 / sent_01` | 목표 한국어 발화 뒤에 비목표 외국어 음성이 이어짐 | 수정 필요 사항으로 기록, 기존 보류 상태 유지 | 목표 발화 종료 시점을 확인한 뒤 후반부 추가 음성 절단 및 해당 원본 기반 파생본 재생성 검토 |

### S004/sent_02 — 선행 무음 처리
- 파일: `spkS004__sent_02__normal_energy.wav`
- 선행 무음은 문장 누락이나 비정상 연장과 구분하여 기록한다.
- 현재 원본 및 파생본은 수정하지 않는다.
- 학습 입력 전처리 단계에서 선행 무음 처리 여부를 검토한다.
- 파일 시작부터 고정 길이로 입력을 절단할 경우,
  선행 무음 때문에 실제 발화 일부가 누락되지 않는지 확인한다.
- 선행 무음 정리와 문장 내부 멈춤 제거는 구분한다.
- 약 1초라는 청취 관찰값을 확정 절단 시점이나 자동 처리 임계값으로 사용하지 않는다.

### S028/sent_01 — 후반부 비목표 음성
- 파일: `spkS028__sent_01__normal_energy.wav`
- 기존 tremor QC 과정에서 발견한 후반부 외국어 추가 음성 문제를
  이번 normal_energy 전수 청취에서도 재확인했다.
- 목표 발화 뒤의 비목표 음성을 제거하는 수정이 필요하다.
- 정확한 절단 시점은 목표 문장의 종결부를 확인한 뒤 결정한다.
- 현재는 발견 및 수정 필요 사항만 기록하며, 원본 절단과 파생본 재생성은 수행하지 않았다.
- 관련 파생본 영향 범위는 기존 tremor QC 기록을 참조한다.

### 연장 검출 평가 시 고려 사항
- 기존에는 느린 정상 화자와 연장 파생본의 구분을 주요 비교 과제로 생각했다.
- 이번 청취를 통해 다음 정상 발화 특성도 연장 검출의 혼동 요인으로
  검토할 필요가 있음을 확인했다.
  1. 자연스럽게 느린 발화
  2. 문장·구절 말미의 늘어짐
  3. 억양 또는 강조에 따른 국소적인 늘어짐
- 위 특성이 있다는 이유만으로 정상 원본을 제외하거나 연장 라벨로 변경하지 않는다.
- 1차 학습 후 해당 정상 구간이 연장으로 오탐되는지 확인한다.
- 혼동이 확인되면 2차 실험에서 비교 사례와 처리 방식을 보완한다.
- 현 단계에서는 새로운 길이 임계값이나 자동 판정 기준을 확정하지 않는다.

---

### S028 파생본 교정 및 최종 교체
- S028/sent_01 원본을 4.4초로 교정.
- 파생본 4개(fade-in/out, slow-normal, tremor) 재생성 및 청취 통과.
- 기존 최종 audio 파일 교체 완료. 백업과 검토본은 보존.
- S028 prolongation 유지, S039 tremor 제외 판정 유지.
- 실행 ID: S028_sent01_rebuild_20261010T030845_539435