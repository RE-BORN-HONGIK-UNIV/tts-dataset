# tts-dataset

고립청년 대면훈련 서비스의 **면접 음성 분석 모델 학습 데이터 생성 코드**.
Typecast TTS API로 정상·이상 발화 음성을 합성하고, 클래스·화자 라벨이 정확한 메타데이터를
함께 기록한다.

서비스 코드가 아니라 **데이터 생성 전용 리포지토리**다.
(서비스 구현은 `backend` / `frontend`, 모델 서빙은 `ai-agent`)

## 파일 이름 규칙

- `src/sN_...` : 데이터를 **만드는** 파이프라인. `N`은 실행 순서(s1 → s2 → s3 → s4).
- `tools/...`  : 데이터를 **검사하는** 도구. 실행 순서에 속하지 않고 필요할 때 반복 실행하므로 번호가 없다.

`s`는 step의 약자다. `01_...`처럼 숫자로 시작하면 파이썬 모듈로 import 할 수 없기 때문에
문자로 시작한다. 나중에 함수를 재사용할 때 `from src.s3_generate_energy import ...` 형태로
그대로 불러 쓸 수 있다.

---

## 무엇을 만드는가

CNN으로 분류할 네 가지 클래스 중 코드로 합성할 수 있는 클래스를 생성한다.

| 클래스 | 뜻 | 만드는 방법 |
|---|---|---|
| `normal` | 정상 발화 | TTS 원본 그대로 (후처리 없음) |
| `prolong` | 연장 (말을 늘여 끄는 것) | 이전 작업분 83개를 가져옴 — 이 리포지토리에서 생성하지 않음 |
| `energy` | 에너지 변동 | TTS 원본 + 연속 게인 엔벨로프 |
| `tremor` | 떨림 | 추후 |

`energy`(에너지 변동)는 면접 중 **자신감 변화로 음량이 한 방향으로 흐르는 현상**이다.
문장이 갈수록 작아지거나(fade out), 작게 시작했다가 커지는(fade in) 경우가 해당한다.

채움말·멈춤 검출은 파이썬 규칙 기반이라 별도 트랙이다. 이 리포지토리는 CNN 학습용
합성 데이터(연장·에너지변동·떨림)와 그 비교 기준인 정상 발화를 다룬다.

---

## 실행 순서

```
pip install -r requirements.txt
cp .env.example .env        # .env 안에 TYPECAST_API_KEY 입력

python src/s1_check_voices.py             # 화자 풀 확정 (크레딧 소모 없음)
python src/s2_generate_normal_pilot.py    # 정상 30개 파일럿
python tools/measure_energy_envelope.py   # 정상군 엔벨로프 분포 측정 (크레딧 소모 없음)
   -> 이 분포를 기준으로 s3의 변동 폭을 확정한다
python src/s3_generate_energy.py          # 에너지 변동 데이터 생성
python tools/measure_energy_envelope.py --dir data/energy/v4   # 정상군과 분리되는지 검증
python src/s4_build_master_metadata.py    # 메타데이터 통합
```

모든 명령은 repo 최상위 폴더(`tts-dataset/`)에서 실행한다.
각 파일 맨 위 주석에 **왜 그렇게 했는지**가 정리돼 있다.

| 파일 | 역할 |
|---|---|
| `src/s1_check_voices.py` | Typecast 화자 목록 조회, 조건 필터, `voice_candidates.csv` 출력 |
| `src/s2_generate_normal_pilot.py` | 정상 30개 생성 + 미리듣기 다운로드 |
| `src/s3_generate_energy.py` | 에너지 변동 클래스 생성 (게인 엔벨로프 방식) |
| `src/s4_build_master_metadata.py` | 배치별 CSV를 클래스별 정렬 마스터 한 장으로 통합 |
| `tools/measure_energy_envelope.py` | wav 폴더의 문장 단위 음량 흐름 측정 (`trend_db`, `slope_db_per_s`, `env_range_db`) |

---

## 설계 원칙

### 1. 화자 수가 데이터 개수보다 중요하다
화자 5명으로 1,200개를 만들면 모델이 그 5명의 목소리를 외운다(speaker overfitting).
그래서 생성 전에 조건에 맞는 화자를 먼저 세고(`s1`), 남녀 균형을 맞춰 뽑는다.
학습/검증/테스트 분할도 **화자 단위**로 한다. 같은 화자가 분할을 넘나들면
성능이 과대평가된다.

### 2. 정상군도 같은 TTS로 만든다
정상군을 실제 녹음으로 구하면 마이크 특성·배경 잡음이 클래스와 완벽하게 상관되어,
모델이 "합성음이냐 실제 녹음이냐"를 학습한다. 같은 화자·같은 문장·같은
`target_lufs`로 생성하면 두 클래스의 차이가 음량 변동 하나로 통제된다.

다만 합성 데이터가 새로운 편향을 도입·증폭할 수 있다는 지적이 있으므로
([Computational Linguistics](https://direct.mit.edu/coli/article/51/1/191/124625/Evaluating-Synthetic-Data-Generation-from-User)),
최종 평가는 실제 사용자 음성으로 해야 한다.

### 3. 문장은 한 번에 합성하고, 변동은 후처리로 준다
문장을 토막내 따로 합성한 뒤 이어붙이면 이음새 아티팩트가 남아, CNN이 에너지 변동이
아니라 그 흔적을 학습한다. 전체를 한 번에 합성해 운율을 보존하고, 변동은 numpy에서
연속 코사인 게인 엔벨로프로 부여한다. 패딩·복사는 쓰지 않는다.

### 4. 라벨은 목표값이 아니라 실측값을 쓴다
부여한 변동 폭(target)과 실제 측정값(measured)은 다르다. TTS 원본 자체에도 음량 기복이
있기 때문이다. **임계값 검토와 라벨링에는 실측값을 쓴다.**

### 5. 측정 지표는 "문장 단위 흐름"을 재야 한다
파일럿(s2)에서 처음 쓴 `swing_db`는 50ms 프레임의 최대/최소 비였다. 결과는 30개 전부
20.8~26.0 dB에 몰렸고, 최대값 25.99 dB는 정의상 천장인 \(20\log_{10}(1/0.05) = 26.02\) dB에
붙어 있었다(무음 제거 기준이 "최대의 5% 미만"이므로 최대/최소 비가 20을 넘을 수 없음).

50ms는 음절보다 짧아서 모음과 자음·단어 사이 틈의 차이를 재게 된다. 즉 이 지표는
"문장이 점점 작아지는가"가 아니라 "음절이 얼마나 출렁이는가"를 쟀고, 에너지 변동을
준 클립도 같은 천장에 붙으므로 클래스를 구분할 수 없다.

그래서 `tools/measure_energy_envelope.py`에서 지표를 새로 정의했다.

| 지표 | 정의 | 용도 |
|---|---|---|
| `trend_db` | 유성 프레임 뒤 25% 평균 dB − 앞 25% 평균 dB | fade_out / fade_in |
| `slope_db_per_s` | (시각, dB)에 맞춘 직선 기울기 | 길이가 다른 클립 비교 |
| `env_range_db` | 0.5초 이동평균으로 평활한 dB 곡선의 최대 − 최소 | swell / dip |

기존 `swing_db`는 `legacy_swing_db`로 함께 계산해 천장 문제를 수치로 남긴다.
합성 사인파 검증에서는 드러나지 않았고 실제 음성 파일럿에서 발견된 문제다.
본 생성 전에 파일럿을 두는 이유다.

### 6. 임계값은 근거 없이 정하지 않는다
정상 발화에도 음량이 줄어드는 경향이 원래 있다. 발화가 진행되며 F0와 강도가 함께
감소하는 declination은 언어 보편적 경향으로 보고되며
([Lieberman 인용, Antwerp](https://repository.uantwerpen.be/docman/irua/0b0ce0/154885.pdf)),
평서문은 종결부 음절에서 dB SPL이 급격히 떨어지는 것이 특징으로 기술된다
([University of Alberta](https://era.library.ualberta.ca/items/a14f3137-5fe8-46bb-a6a0-dbf7d48c1a8f/download/61497a8d-33b7-477d-93a2-15dfabac3338)).
발화 말/비말 음절의 강도 비는 1.33~5.76 dB로 보고됐다
([PMC3212410](https://pmc.ncbi.nlm.nih.gov/articles/PMC3212410/)).

파일럿 정상군(s2 정의 `trend_db`)도 평균 −2.46 dB, 범위 −7.00 ~ +0.72 dB,
30개 중 27개(90%)가 뒤로 갈수록 작아져 declination과 같은 방향을 보였다.

**단, 선행 연구 수치를 그대로 임계값으로 쓸 수는 없다.** 선행 연구는 종결 음절 대
비종결 음절을 비교하고, 우리 지표는 문장 앞 1/4과 뒤 1/4을 비교한다. 구간 정의가 달라
직접 전이가 불가능하다. 그래서 정상군 분포를 실측한 뒤 임계값을 정한다.
현재 `s3`의 `SWING_DB_RANGE = (10.0, 20.0)`과 측정 도구의 `VOICED_RANGE_DB = 30`,
`SMOOTH_SEC = 0.5`는 **잠정값**이다.

### 7. 화자는 한국어 합성 음성을 듣고 판정한다
태그(`use_cases`)로는 부적합 화자를 걸러낼 수 없었다. 파일럿 10명 판정 결과
(2026-09-24):

| 화자 | 태그 | 판정 |
|---|---|---|
| Buttaguy | Conversational | 제외 — 캐릭터형 발성 |
| Jabbaba | TikTok/Reels/Shorts | 제외 — 캐릭터형 발성 |
| Ravi, Zoey | E-learning, Radio/Podcast 등 | 사용 — 미리듣기는 영어였으나 한국어 합성은 자연스러움 |
| 나머지 6명 | | 사용 |

미리듣기 음원은 Typecast가 정한 샘플 문장이라 언어가 다를 수 있다.
**판정은 우리가 한국어로 합성한 음성으로 한다.**

---

## 데이터 구조

코드와 데이터가 한 repo 폴더 안에 있고, `data/`만 GitHub에서 제외된다.
코드는 경로를 절대경로(`C:\...`)가 아니라 "코드 파일 기준 한 칸 위의 `data/`"로
찾기 때문에 repo 폴더를 어디로 옮겨도 그대로 동작한다.

```
tts-dataset/
  src/                          생성 파이프라인           ← GitHub
  tools/                        검사 도구                ← GitHub
  README.md, requirements.txt, .gitignore               ← GitHub
  .env                          API 키                  ← 제외
  data/                         음성·명단·메타데이터     ← 제외 (.gitignore)
    voice_candidates.csv        s1 출력. 생성 스크립트가 이 명단에서 화자를 뽑는다
    prolong/                    연장 83개 (이전 작업분)
      audio/  metadata.csv
    normal/v4/                  정상 클래스
    energy/v4/                  에너지 변동
    tremor/                     떨림 (추후)
    _pilot_normal30/            측정용 파일럿 (학습 제외)
      audio/  _previews/  metadata.csv  envelope_metrics.csv
    master_metadata.csv         s4 출력
    by_class/{class}.csv        s4 출력, 클래스별 분리본
```

연장 83개는 이전 작업 폴더(`C:\reborn_tts_data\organized\`)에서 화자 클러스터링을
마친 결과를 복사해 온 것이다. 원본은 이전 폴더에 보관한다.

`metadata.csv`는 모든 폴더에서 **같은 이름**을 쓴다. 어느 클래스인지는 폴더로
구분한다. s4는 이 이름으로 파일을 찾아 모으고, s3는 이 파일들에서 이미 쓰인
화자 번호를 읽어 겹치지 않는 다음 번호를 매긴다.

### 밑줄 규칙
**이름이 밑줄(`_`)로 시작하는 폴더는 마스터 통합에서 제외된다.**
폴더 이름만 보고 학습용인지 실험용인지 구분하기 위한 장치다.
제외된 폴더도 CSV는 남긴다. 임계값을 정한 근거 자료라 발표·논문에 필요하다.

### 파일명 규칙
```
{class}_{speaker_id}_{gender}_{seq}.wav

energy_spk009_f_001.wav     본 생성
normal_spkP01_m_003.wav     파일럿 (P를 붙여 본 생성 번호와 구분)
```
`seq`는 화자별 연번이라 폴더를 정렬만 해도 화자별 개수가 보인다.
API로 생성하면 `voice_id`를 이미 알고 있으므로 화자 클러스터링이 필요 없고,
화자 라벨이 추정이 아니라 정확하다.

### 메타데이터
배치 폴더마다 `metadata.csv`를 두고 `s4`가 이를 모아 마스터를 만든다.
마스터는 **항상 다시 만드는 파일**이다. 직접 수정하지 않고, 배치 CSV를 고친 뒤
`s4`를 다시 실행한다.

공통 열: `new_filename, old_filename, class, speaker_id, gender, speaker_f0_hz,
gender_flag, source, voice_name, text_script, qc_pass, cluster_raw, is_representative`

부가 열: `batch, pattern, swing_db_target, swing_db_measured, trend_db, duration_sec,
sample_rate, voice_id, base_lufs, seed, use_cases, lang_flag, script_version, generated_at`

`s4`는 클래스 x 화자 교차표를 출력한다. 한 화자가 한 클래스에만 등장하면
모델이 음향 패턴이 아니라 "누구 목소리인가"를 학습할 수 있으므로(speaker leakage)
경고를 띄운다.

---

## 알려진 한계

- **Typecast API에 언어 정보가 없다.** 응답에 language / locale / nationality 필드가
  없어 API로 외국어 화자를 구분할 수 없다
  ([List Voices](https://typecast.ai/docs/api-reference/voices/list-voices),
  [Get Voice Details](https://typecast.ai/docs/api-reference/voices/get-voice-details)).
  한국어로 합성한 음성을 듣고 판정한다(설계 원칙 7).
- **캐릭터형 발성도 태그로 걸러지지 않는다.** 파일럿 10명 중 2명이 해당됐다.
  본 생성 전에 청취 스크리닝이 필요하다.
- **공식 문서와 실제 `use_cases` 태그 이름이 다르다.** 예: 문서 `Podcast` → 실제 `Radio/Podcast`.
  그래서 코드는 소문자 부분일치로 비교한다.
- **길이 조건에 걸려 버려지는 합성도 크레딧이 든다.** 파일럿에서 3.0초 기준으로 57회 중
  27회가 버려졌다(대부분 2.2~2.9초). 최소 길이를 2.5초로 완화했다.
- **게인 엔벨로프는 음량만 바꾼다.** 실제로 작게 말할 때 나타나는 숨소리·발성 방식 변화는
  재현되지 않는다. 논문 limitation에 명시할 항목이다.
- **TTS 음성들이 같은 백본을 공유한다.** 화자 수백 명이 독립적인 음향 도메인 수백 개를
  의미하지 않는다.

---

## 요구 사항

Python 3.9+, `typecast-python`, `requests`, `python-dotenv`, `soundfile`, `numpy`
API 키는 `.env`의 `TYPECAST_API_KEY`로 읽는다. `.env`는 `.gitignore`에 포함돼 있다.
