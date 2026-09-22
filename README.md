# tts-dataset

고립청년 대면훈련 서비스의 **면접 음성 분석 모델 학습 데이터를 만드는 코드**입니다.
Typecast TTS API로 이상 발화 음성을 합성하고, 클래스·화자 라벨이 정확한 메타데이터를
함께 기록합니다.

서비스 코드가 아니라 **데이터 생성 전용 리포지토리**입니다.
(서비스 구현은 `backend` / `frontend`, 모델 서빙은 `ai-agent` 쪽)

## 파일 이름 규칙

`sN_` 접두어는 파이프라인 실행 순서입니다(s1 → s2 → s3 → s4).
숫자로 시작하지 않는 이유는 `01_...` 형태는 파이썬 모듈로 import 할 수 없기 때문입니다.
나중에 함수를 재사용하게 되어도 `from src.s3_generate_energy import measure_rms_swing_db`
처럼 그대로 불러 쓸 수 있습니다.

---

## 무엇을 만드는가

CNN으로 분류할 네 가지 클래스 중, 코드로 합성할 수 있는 클래스를 생성합니다.

| 클래스 | 뜻 | 만드는 방법 |
|---|---|---|
| `normal` | 정상 발화 | TTS 원본 그대로 (후처리 없음) |
| `prolong` | 연장 (말을 늘여 끄는 것) | 수작업 녹음 — 이 리포지토리 범위 밖 |
| `energy` | 에너지 변동 | TTS 원본 + 연속 게인 엔벨로프 |
| `tremor` | 떨림 | 추후 |

`energy`(에너지 변동)는 면접 중 **자신감 변화로 음량이 한 방향으로 흐르는 현상**을 말합니다.
문장이 갈수록 작아지거나(fade out), 작게 시작했다가 커지는(fade in) 경우입니다.

채움말·멈춤 검출은 파이썬 규칙 기반이라 별도 트랙이며, 이 리포지토리는 CNN 학습용
합성 데이터(연장·에너지변동·떨림) 쪽을 다룹니다.

---

## 실행 순서

```
pip install -r requirements.txt
cp .env.example .env        # .env 안에 TYPECAST_API_KEY 입력

python src/s1_check_voices.py             # 화자 풀 확정 (크레딧 소모 없음)
python src/s2_generate_normal_pilot.py    # 정상 30개 + 변동 폭 실측
   -> 여기서 나온 분포를 보고 s3의 SWING_DB_RANGE를 확정한다
python src/s3_generate_energy.py          # 에너지 변동 데이터 생성
python src/s4_build_master_metadata.py    # 메타데이터 통합
```

각 스크립트 맨 위 주석에 **왜 그렇게 했는지**가 정리돼 있습니다.

| 파일 | 역할 |
|---|---|
| `src/s1_check_voices.py` | Typecast 화자 목록 조회, 조건 필터, `voice_candidates.csv` 출력 |
| `src/s2_generate_normal_pilot.py` | 정상 30개 생성 + `swing_db` / `trend_db` 분포 측정 + 미리듣기 다운로드 |
| `src/s3_generate_energy.py` | 에너지 변동 클래스 생성 (게인 엔벨로프 방식) |
| `src/s4_build_master_metadata.py` | 배치별 CSV를 클래스별 정렬 마스터 한 장으로 통합 |

---

## 설계 원칙

### 1. 화자 수가 데이터 개수보다 중요하다
화자 5명으로 1,200개를 만들면 모델이 그 5명의 목소리를 외웁니다(speaker overfitting).
그래서 생성 전에 조건에 맞는 화자를 먼저 세고(`s1`), 남녀 균형을 맞춰 뽑습니다.
학습/검증/테스트 분할도 **화자 단위**로 해야 합니다. 같은 화자가 분할을 넘나들면
성능이 과대평가됩니다.

### 2. 정상군도 같은 TTS로 만든다
정상군을 실제 녹음으로 구하면 마이크 특성·배경 잡음이 클래스와 완벽하게 상관되어,
모델이 "합성음이냐 실제 녹음이냐"를 학습해버립니다. 같은 화자·같은 문장·같은
`target_lufs`로 생성하면 두 클래스의 차이가 음량 변동 하나로 통제됩니다.

다만 합성 데이터가 새로운 편향을 도입·증폭할 수 있다는 지적이 있으므로
([Computational Linguistics](https://direct.mit.edu/coli/article/51/1/191/124625/Evaluating-Synthetic-Data-Generation-from-User)),
최종 평가는 실제 사용자 음성으로 해야 합니다.

### 3. 문장은 한 번에 합성하고, 변동은 후처리로 준다
문장을 토막내 따로 합성한 뒤 이어붙이면 이음새 아티팩트가 남아 CNN이 에너지 변동이
아니라 그 흔적을 학습합니다. 전체를 한 번에 합성해 운율을 보존하고, 변동은 numpy에서
연속 코사인 게인 엔벨로프로 부여합니다. 패딩·복사는 쓰지 않습니다.

### 4. 라벨은 목표값이 아니라 실측값을 쓴다
부여한 변동 폭(`swing_db_target`)보다 실제 측정값(`swing_db_measured`)이 보통 큽니다.
엔벨로프 지터와 TTS 원본 자체의 음량 기복이 더해지기 때문입니다.
**임계값 검토와 라벨링에는 `swing_db_measured`를 씁니다.**

### 5. 임계값은 근거 없이 정하지 않는다
정상 발화에도 음량이 줄어드는 경향이 원래 있습니다. 발화가 진행되며 F0와 강도가 함께
감소하는 declination은 언어 보편적 경향으로 보고되며
([Lieberman 인용, Antwerp](https://repository.uantwerpen.be/docman/irua/0b0ce0/154885.pdf)),
평서문은 종결부 음절에서 dB SPL이 급격히 떨어지는 것이 특징으로 기술됩니다
([University of Alberta](https://era.library.ualberta.ca/items/a14f3137-5fe8-46bb-a6a0-dbf7d48c1a8f/download/61497a8d-33b7-477d-93a2-15dfabac3338)).
발화 말/비말 음절의 강도 비는 1.33~5.76 dB로 보고됐습니다
([PMC3212410](https://pmc.ncbi.nlm.nih.gov/articles/PMC3212410/)).

**단, 이 수치를 그대로 쓸 수 없습니다.** 선행 연구는 종결 음절 대 비종결 음절을
비교하는데, 우리 지표는 클립 전체에서 가장 큰 구간과 가장 작은 구간의 차이입니다.
측정 단위와 구간 정의가 달라 직접 전이가 불가능합니다.
그래서 `s2`로 정상군 분포를 실측한 뒤 임계값을 정합니다.
현재 `s3`의 `SWING_DB_RANGE = (10.0, 20.0)`은 **잠정값**입니다.

---

## 데이터 구조

```
C:\reborn_tts_data\
  voice_candidates.csv            s1 출력. 모든 생성 스크립트가 이 파일을 읽는다
  prolongation\real\              수작업 연장 녹음
  organized\                      화자 클러스터링 결과 (수작업 파일용)
  normal\v4\                      정상 클래스 생성분
  energy\v4\                      에너지 변동 생성분
  _pilot_normal30\                측정용 파일럿 (학습 제외)
  _test_batch1\ _test_batch2\ _test_batch3\    과거 코드 테스트 (학습 제외)
  master_metadata.csv             s4 출력
  by_class\{class}.csv            s4 출력, 클래스별 분리본
```

### 밑줄 규칙
**이름이 밑줄(`_`)로 시작하는 폴더는 마스터 통합에서 제외됩니다.**
폴더 이름만 보고 학습용인지 실험용인지 구분할 수 있게 하는 장치입니다.
제외된 폴더도 CSV는 남깁니다. 임계값을 정한 근거 자료라 발표·논문에 필요합니다.

### 파일명 규칙
```
{class}_{speaker_id}_{gender}_{seq}.wav

energy_spk009_f_001.wav     본 생성
normal_spkP01_m_003.wav     파일럿 (P를 붙여 본 생성 번호와 구분)
```
`seq`는 화자별 연번이라 폴더를 정렬만 해도 화자별 개수가 보입니다.
API로 생성하면 `voice_id`를 이미 알고 있으므로 화자 클러스터링이 필요 없고,
화자 라벨이 추정이 아니라 정확합니다.

### 메타데이터
배치 폴더마다 `metadata.csv`를 두고, `s4`가 이를 모아 마스터를 만듭니다.
마스터는 **항상 다시 만드는 파일**입니다. 직접 수정하지 말고 배치 CSV를 고친 뒤
`s4`를 다시 실행하세요.

공통 열: `new_filename, old_filename, class, speaker_id, gender, speaker_f0_hz,
gender_flag, source, voice_name, text_script, qc_pass, cluster_raw, is_representative`

부가 열: `batch, pattern, swing_db_target, swing_db_measured, trend_db, duration_sec,
sample_rate, voice_id, base_lufs, seed, use_cases, lang_flag, script_version, generated_at`

`s4`는 클래스 x 화자 교차표를 출력합니다. 한 화자가 한 클래스에만 등장하면
모델이 음향 패턴이 아니라 "누구 목소리인가"를 외울 수 있으므로(speaker leakage)
경고를 띄웁니다.

---

## 알려진 한계

- **Typecast API에 언어 정보가 없습니다.** 응답에 language / locale / nationality 필드가
  없어 일본어·중국어 화자가 같은 목록에 섞여도 API로 구분할 수 없습니다
  ([List Voices](https://typecast.ai/docs/api-reference/voices/list-voices),
  [Get Voice Details](https://typecast.ai/docs/api-reference/voices/get-voice-details)).
  `s2`에서 미리듣기 음원을 내려받아 귀로 확인하고 `lang_flag` 열에 `ko` / `foreign`을 기입합니다.
- **공식 문서와 실제 `use_cases` 태그 이름이 다릅니다.** 예: 문서 `Podcast` → 실제 `Radio/Podcast`.
  그래서 코드는 소문자 부분일치로 비교합니다.
- **게인 엔벨로프는 음량만 바꿉니다.** 실제로 작게 말할 때 나타나는 숨소리·발성 방식 변화는
  재현되지 않습니다. 논문 limitation에 명시할 항목입니다.
- **TTS 음성들이 같은 백본을 공유합니다.** 화자 600명이 독립적인 음향 도메인 600개를
  의미하지 않습니다.

---

## 요구 사항

Python 3.9+, `typecast-python`, `requests`, `python-dotenv`, `soundfile`, `numpy`
API 키는 `.env`의 `TYPECAST_API_KEY`로 읽습니다. `.env`는 `.gitignore`에 포함돼 있습니다.
