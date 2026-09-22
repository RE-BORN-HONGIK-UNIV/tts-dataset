"""
=============================================================================
s3_generate_energy.py  —  에너지 변동(energy) 클래스 데이터 생성
=============================================================================

!! 주의: 변동 폭(SWING_DB_RANGE)은 02 파일럿 실측 결과를 반영해 확정해야 한다.
         현재 값은 근거가 확정되지 않은 잠정값이다. 아래 THRESHOLD NOTE 참고.

[이 파일의 역할]
  면접 중 자신감 변화로 음량이 한 방향으로 흐르는 발화를 합성한다.
      fade_out : 자신감이 꺾여 문장이 갈수록 작아짐
      fade_in  : 작게 시작했다가 말하면서 커짐
  보조 변이로 swell(작다-크다-작다), dip(크다-작다-크다)을 소수 섞는다.
  보조 변이를 섞는 이유는 모델이 "단조 기울기"만 보고 판단하도록 과적합되는
  것을 막기 위해서다.

[이전 버전(v4)에서 무엇을 왜 바꿨는가]
  v4 는 문장을 4토막으로 쪼개 각 조각을 다른 target_lufs 로 따로 합성하고
  크로스페이드로 이어붙였다. 문제가 네 가지였다.
    1) 조각별 합성이라 운율이 끊기고, 이음새에 인위적 흔적이 남았다.
       -> CNN 이 '에너지 변동'이 아니라 '이음새 아티팩트'를 학습할 위험.
    2) 조각마다 개별 정규화되므로 음량이 계단식으로 점프했다.
       실제 발화의 연속적인 변화와 다르다.
    3) 조용한 구간에 whisper 프리셋을 섞어 문장 중간에 음색이 바뀌었다.
       -> 한 클립 안에서 화자 동일성이 깨졌다.
    4) 길이 보정에 reflect 패딩을 써서 짧은 클립을 거울 복사했다.
       -> 같은 음성이 되풀이되는 오염 데이터가 만들어졌다.

  현재 버전의 해결:
    - 문장 전체를 한 번에 합성한다(운율 보존, 이음새 없음).
    - 에너지 변동은 합성 후 numpy 에서 '연속 게인 엔벨로프'로 부여한다.
      코사인 곡선을 써서 미분 불연속(뚝 끊기는 느낌)이 없다.
    - 패딩을 쓰지 않는다. 무음 트림 후 3~5초 범위를 벗어나면 그 클립을 버리고
      다른 문장으로 다시 시도한다.
    - 라벨이 정확하다. 우리가 부여한 변동 폭(swing_db_target)과
      실제 측정값(swing_db_measured)을 둘 다 기록한다.

[후처리 게인 엔벨로프가 타당한 이유]
  CNN 이 학습할 특징은 프레임별 에너지 변화다. 게인 엔벨로프는 그 특징을
  직접 통제하므로 라벨과 신호가 정확히 일치한다. 정상군과 원본이 동일하고
  차이가 엔벨로프 하나뿐이므로, 통제 실험으로서 교란 변수가 없다.
  (한계: 실제로 작게 말할 때 나타나는 숨소리 등 음색 변화는 재현되지 않는다.
   이 한계는 논문 limitation 에 명시할 항목이다.)

[측정값이 목표값보다 크게 나오는 이유]
  swing_db_measured 가 swing_db_target 보다 보통 크다. 두 요인 때문이다.
    - 엔벨로프에 자연스러움을 위해 넣은 미세 지터(+-약 1.4 dB)
    - TTS 원본 자체가 이미 갖고 있는 음량 기복
  라벨링과 임계값 검토에는 반드시 swing_db_measured 를 쓴다.

[입력]  voice_candidates.csv  (s1 의 출력)
[출력]  energy/v4/audio/energy_spk001_f_001.wav ...
        energy/v4/metadata.csv

[파일명 규칙]  {class}_{speaker_id}_{gender}_{seq}.wav
  seq 는 화자별 연번이다. 폴더를 정렬만 해도 화자별 개수가 눈에 보인다.
  API 로 생성하면 voice_id 를 이미 알고 있으므로 화자 클러스터링이 필요 없고,
  화자 라벨이 추정이 아니라 정확하다. (수작업 파일은 클러스터링으로 추정해야 한다)

[실행]
  pip install typecast-python python-dotenv soundfile numpy
  python s3_generate_energy.py
=============================================================================
"""

import os
import io
import csv
import time
import random

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from typecast import Typecast
from typecast.models import TTSRequest, Output, LanguageCode

load_dotenv()

# ------------------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------------
# --- 저장 위치 (기존 reborn_tts_data 구조에 맞춤) --------------------
# 과거 테스트 결과물은 밑줄을 붙여 보관한다(_test_batch1~3). 마스터 통합에서 제외된다.
# 이번 생성분은 energy/v4/ 에 들어간다.
DATA_ROOT = r"C:\reborn_tts_data"
BATCH_NAME = "v4"   # energy/v4/ 에 저장. 기존 _test_batch1~3 과 구분된다
OUTPUT_DIR = os.path.join(DATA_ROOT, "energy", BATCH_NAME)
CANDIDATES_CSV = os.path.join(DATA_ROOT, "voice_candidates.csv")  # s1 의 출력
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
META_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

CLASS_NAME = "energy"     # metadata의 class 열 값 (prolong / energy / tremor / normal)
SOURCE_TAG = "tts"

# speaker_id 시작 번호.
#  기존 organized\metadata.csv 에서 이미 spk001~spkNNN 을 쓰고 있으면
#  겹치지 않게 그 다음 번호부터 시작해야 한다. 실행 시 자동 감지도 시도한다.
SPEAKER_ID_START = 1
EXISTING_META_FOR_SPK = os.path.join(DATA_ROOT, "organized", "metadata.csv")

TARGET_COUNT = 200          # 1차 생성 개수
NUM_VOICES_TO_USE = 8       # 중복 없이 확보할 청년 화자 수
RANDOM_SEED = 20260922      # 재현성 확보

MIN_DURATION_SEC = 3.0
MAX_DURATION_SEC = 5.0

BASE_LUFS = -18.0           # TTS 원본은 항상 동일 기준으로 받아온다 (변동은 후처리로)

# --- THRESHOLD NOTE : 아래 값은 학술 근거가 확정되지 않은 잠정값이다 -------
#
# 정의: 클립 내 구간 RMS 의 최대-최소 차이(dB). 무음 프레임은 제외한다.
#
# 참고할 수 있는 선행 연구
#   - 발화가 진행되며 F0 와 강도가 함께 감소하는 declination 은 언어 보편적
#     경향으로 보고된다 (Lieberman 인용).
#     https://repository.uantwerpen.be/docman/irua/0b0ce0/154885.pdf
#   - 평서문은 종결부 음절에서 dB SPL 이 급격히 떨어지는 것이 특징으로 기술된다.
#     https://era.library.ualberta.ca/items/a14f3137-5fe8-46bb-a6a0-dbf7d48c1a8f/download/61497a8d-33b7-477d-93a2-15dfabac3338
#   - 발화 말/비말 음절의 강도 비는 1.33~5.76 dB 범위로 보고됐다.
#     https://pmc.ncbi.nlm.nih.gov/articles/PMC3212410/
#
# 이 값들을 그대로 임계값으로 쓸 수 없는 이유 (반드시 논문에 명시할 부분)
#   위 수치는 '종결 음절 대 비종결 음절'의 비교다. 우리 지표는 '클립 전체에서
#   가장 큰 구간과 가장 작은 구간의 차이'이므로 측정 단위와 구간 정의가 다르다.
#   따라서 수치를 직접 전이할 수 없다. 우리 파이프라인에서 실측해야 한다.
#
# 확정 절차
#   1) s2_generate_normal_pilot.py 로 정상군 swing_db 분포를 실측한다.
#   2) 정상군 상위 백분위(예: 95백분위)를 확인한다.
#   3) 두 분포가 겹치지 않도록 그보다 충분히 큰 값에서 하한을 잡는다.
#   -> 이 절차를 거치기 전까지 아래 값은 잠정값이며 논문에 그대로 쓸 수 없다.
SWING_DB_RANGE = (10.0, 20.0)   # 잠정값 (파일럿 실측 반영 대기)
# ------------------------------------------------------------------

# 엔벨로프 패턴 비중
#  핵심 타깃: 면접 중 자신감이 꺾여 문장이 갈수록 작아지는 경우(fade_out),
#            작게 시작했다가 말하면서 커지는 경우(fade_in).
#  swell/dip은 보조 변이로 소수만 섞어 모델이 '단조 기울기'에만 과적합되지 않게 함.
PATTERN_WEIGHTS = {
    "fade_out": 0.40,
    "fade_in": 0.40,
    "swell": 0.10,
    "dip": 0.10,
}


def build_pattern_schedule(total: int):
    """비중대로 패턴 목록을 정확히 total개 만들어 섞어서 반환."""
    sched = []
    for name, w in PATTERN_WEIGHTS.items():
        sched += [name] * int(round(total * w))
    while len(sched) < total:
        sched.append("fade_out")
    sched = sched[:total]
    random.shuffle(sched)
    return sched

# 제외 태그 (소문자 부분일치). s1_check_voices.py 와 동일하게 유지해야
# 정상군과 에너지군의 화자 풀이 일치한다.
EXCLUDE_USE_CASE_KEYWORDS = ("rapper", "anime", "game")

SILENCE_TRIM_DB = 40.0      # 피크 대비 -40 dB 이하 구간은 앞뒤 무음으로 보고 잘라냄
MAX_RETRY = 3
RETRY_SLEEP_SEC = 2.0

os.makedirs(AUDIO_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

client = Typecast()

TEXT_POOL = [
    "저는 대학교에서 컴퓨터공학을 전공했고 다양한 프로젝트를 진행했습니다.",
    "가장 힘들었던 순간은 팀 프로젝트 마감 직전이었는데 그 과정에서 많은 것을 배웠습니다.",
    "이 회사에 지원한 이유는 성장 가능성이 크다고 느꼈기 때문입니다.",
    "인턴 경험을 통해 실무 감각을 익힐 수 있었습니다.",
    "저는 새로운 기술을 배우는 것을 좋아하고 꾸준히 공부하는 습관이 있습니다.",
    "팀원들과 소통하면서 문제를 함께 해결한 경험이 있습니다.",
    "저의 강점은 문제를 끝까지 파고드는 끈기이고 이는 여러 프로젝트에서 드러났습니다.",
    "졸업 프로젝트로 인공지능 관련 서비스를 개발했습니다.",
    "저는 항상 계획을 세우고 체계적으로 일을 진행합니다.",
    "동아리 활동을 통해 리더십을 기를 수 있었습니다.",
    "고객의 입장에서 생각하는 것을 가장 중요하게 여깁니다.",
    "저는 압박감 속에서도 침착하게 대처하는 편입니다.",
    "다양한 프로젝트 경험이 저의 강점이라고 생각합니다.",
    "학교 수업 외에도 스터디를 통해 실력을 쌓았습니다.",
    "제 목표는 이 분야의 전문가가 되는 것입니다.",
    "저는 책임감을 가지고 맡은 일을 끝까지 해냅니다.",
    "협업 프로젝트에서 의견 차이가 생겼을 때 대화를 통해 조율했습니다.",
    "우선순위를 정해서 일을 처리하는 습관이 있습니다.",
    "작은 실수도 놓치지 않으려고 항상 다시 확인합니다.",
    "동료들과의 신뢰를 가장 중요한 가치로 생각합니다.",
    "어려운 문제를 만나면 오히려 더 집중하게 됩니다.",
    "다른 사람의 의견을 경청하는 태도를 중요하게 생각합니다.",
    "시간 관리를 철저히 해서 마감을 지키는 편입니다.",
    "문제 상황에서 침착하게 원인을 분석하려고 합니다.",
    "목표 달성을 위해 세부 계획을 세우는 편입니다.",
    "팀 프로젝트에서는 항상 마감일을 기준으로 역산해서 일정을 짭니다.",
    "실수를 했을 때는 바로 인정하고 개선점을 찾습니다.",
    "다양한 의견을 듣고 최선의 결정을 내리려고 노력합니다.",
    "이 회사의 서비스를 오랫동안 관심 있게 지켜봤습니다.",
    "배운 것을 실제 프로젝트에 적용해보는 걸 즐깁니다.",
    "데이터를 기반으로 의사결정을 하는 습관이 있습니다.",
    "문제가 생기면 원인을 먼저 파악하고 해결책을 찾습니다.",
    "협업 툴을 활용해서 효율적으로 일정을 관리합니다.",
    "어려운 상황일수록 냉정하게 판단하려고 합니다.",
    "제 전공 지식을 실무에 접목시키고 싶습니다.",
    "팀원 간의 갈등을 중재한 경험이 있습니다.",
    "자기 개발을 위해 꾸준히 강의를 듣고 있습니다.",
    "목표를 이루기 위해 작은 습관부터 바꿔나갔습니다.",
    "제가 세운 계획이 틀렸을 때는 빠르게 수정합니다.",
    "다양한 관점에서 문제를 바라보려고 노력합니다.",
    "실무 경험을 쌓기 위해 여러 프로젝트에 참여했습니다.",
    "새로운 기술 트렌드를 놓치지 않으려고 항상 찾아봅니다.",
    "갑작스러운 변화에도 유연하게 대처하려고 합니다.",
    "팀원들의 강점을 파악해서 역할을 나누는 걸 좋아합니다.",
    "어려운 개념도 끝까지 파고들어 이해하려고 합니다.",
    "제 의견을 논리적으로 전달하는 연습을 꾸준히 했습니다.",
    "실패한 프로젝트에서도 배울 점을 찾으려고 합니다.",
    "협업할 때는 항상 상대방의 입장을 먼저 생각합니다.",
    "예상치 못한 문제가 생겨도 당황하지 않으려고 합니다.",
    "피드백을 받으면 바로 반영해서 개선하려고 합니다.",
    "회사의 비전과 제 목표가 잘 맞는다고 느꼈습니다.",
    "주도적으로 문제를 찾아 해결한 경험이 있습니다.",
    "팀 전체의 성과를 위해 제 역할을 다하려고 합니다.",
    "어떤 과제든 마감 전에 여유 있게 끝내려고 합니다.",
]


# ------------------------------------------------------------------
# 1. 화자 선정: 청년(young_adult) + 랩 계열 제외 + 중복 없이
# ------------------------------------------------------------------
def load_candidate_voices():
    """
    s1_check_voices.py 가 만든 voice_candidates.csv 에서 조건 통과 화자를 읽는다.
    API 를 매번 다시 조회하지 않는 이유:
      - 어떤 화자 풀로 만든 데이터인지 기록이 남는다(재현성).
      - 정상군 생성과 완전히 같은 풀을 쓰게 되어 비교가 성립한다.
      - 플랫폼이 화자를 추가/삭제해도 이미 만든 데이터의 근거가 흔들리지 않는다.
    """
    if not os.path.exists(CANDIDATES_CSV):
        raise SystemExit(f"[오류] {CANDIDATES_CSV} 가 없습니다. s1_check_voices.py 를 먼저 실행하세요.")

    with open(CANDIDATES_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    def ok(r):
        if (r.get("excluded_by") or "").strip():          # s1 단계에서 이미 제외됨
            return False
        tags = (r.get("use_cases") or "").lower()
        return not any(kw in tags for kw in EXCLUDE_USE_CASE_KEYWORDS)

    kept = [r for r in rows if ok(r)]
    print(f"[후보] 전체 {len(rows)}명 -> 조건 통과 {len(kept)}명 "
          f"(제외 태그: {list(EXCLUDE_USE_CASE_KEYWORDS)})")
    return kept


def pick_young_voices(n: int):
    """
    후보에서 남/여 균형을 맞춰 중복 없이 n명을 뽑는다.
    성별 균형을 맞추는 이유: 성별에 따라 기본 주파수와 강도 특성이 달라서,
    한쪽으로 쏠리면 모델이 성별 특징을 클래스 신호로 오인할 수 있다.
    """
    cands = load_candidate_voices()
    males = [c for c in cands if c.get("gender") == "male"]
    females = [c for c in cands if c.get("gender") == "female"]
    random.shuffle(males)
    random.shuffle(females)

    half = n // 2
    picked = males[:half] + females[:n - half]
    if len(picked) < n:
        print(f"[경고] 요청 {n}명 중 {len(picked)}명만 확보 "
              f"(남 {len(males)} / 여 {len(females)})")

    random.shuffle(picked)
    out, seen = [], set()
    for c in picked:
        if c["voice_id"] in seen:
            continue
        seen.add(c["voice_id"])
        out.append({
            "voice_id": c["voice_id"],
            "name": c.get("voice_name", ""),
            "gender": "f" if c.get("gender") == "female" else "m",
            "use_cases": c.get("use_cases", ""),
        })
    return out


def next_speaker_number() -> int:
    """
    기존 organized/metadata.csv 에 쓰인 spkNNN 을 읽어 그 다음 번호를 반환.
    파일이 없으면 SPEAKER_ID_START 를 쓴다.
    (같은 성우를 다시 뽑았다면 voice_name 으로 매칭해 같은 번호를 재사용한다.)
    """
    if not os.path.exists(EXISTING_META_FOR_SPK):
        return SPEAKER_ID_START
    used, name_map = [], {}
    with open(EXISTING_META_FOR_SPK, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sid = (r.get("speaker_id") or "").strip()
            if sid.startswith("spk") and sid[3:].isdigit():
                used.append(int(sid[3:]))
                nm = (r.get("voice_name") or "").strip()
                if nm:
                    name_map[nm] = sid
    next_no = (max(used) + 1) if used else SPEAKER_ID_START
    print(f"[speaker_id] 기존 메타에서 최대 spk{max(used):03d} 감지 -> spk{next_no:03d} 부터 부여"
          if used else f"[speaker_id] 기존 메타 없음 -> spk{next_no:03d} 부터 부여")
    return next_no, name_map


def assign_speaker_ids(voices):
    """뽑힌 화자들에게 spkNNN 을 부여. 기존 메타에 같은 성우가 있으면 번호 재사용."""
    res = next_speaker_number()
    start, name_map = res if isinstance(res, tuple) else (res, {})
    n = start
    for v in voices:
        reused = name_map.get(v["name"])
        if reused:
            v["speaker_id"] = reused
            print(f"  {v['name']}: 기존 {reused} 재사용")
        else:
            v["speaker_id"] = f"spk{n:03d}"
            n += 1
    return voices


def estimate_f0_hz(y: np.ndarray, sr: int) -> float:
    """자기상관 기반 중위 F0 추정 (gender 검증용 기록. 성별 자체는 API 값을 신뢰)."""
    frame, hop = int(sr * 0.04), int(sr * 0.02)
    lo, hi = int(sr / 400), int(sr / 70)          # 70~400 Hz 탐색
    vals = []
    for i in range(0, max(0, len(y) - frame), hop):
        seg = y[i:i + frame]
        if np.sqrt(np.mean(seg ** 2)) < 1e-3:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
        if len(ac) <= hi or ac[0] <= 0:
            continue
        lag = int(np.argmax(ac[lo:hi])) + lo
        if ac[lag] / ac[0] > 0.3:
            vals.append(sr / lag)
    return round(float(np.median(vals)), 1) if vals else 0.0

# ------------------------------------------------------------------
# 2. TTS 합성 (문장 전체를 한 번에)
# ------------------------------------------------------------------
def synthesize(text: str, voice_id: str):
    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            res = client.text_to_speech(TTSRequest(
                text=text,
                model="ssfm-v30",
                voice_id=voice_id,
                language=LanguageCode.KOR,
                output=Output(
                    target_lufs=BASE_LUFS,
                    audio_pitch=0,
                    audio_tempo=1.0,
                    audio_format="wav",
                ),
            ))
            audio, sr = sf.read(io.BytesIO(res.audio_data))
            if audio.ndim > 1:                  # 스테레오면 모노로
                audio = audio.mean(axis=1)
            return audio.astype(np.float32), sr
        except Exception as e:                  # 레이트리밋/일시 장애 대응
            last_err = e
            print(f"    [재시도 {attempt}/{MAX_RETRY}] {type(e).__name__}: {e}")
            time.sleep(RETRY_SLEEP_SEC * attempt)
    raise RuntimeError(f"TTS 실패: {last_err}")


# ------------------------------------------------------------------
# 3. 전처리: 앞뒤 무음 트림 + 길이 판정
# ------------------------------------------------------------------
def trim_silence(y: np.ndarray, sr: int, top_db: float = SILENCE_TRIM_DB) -> np.ndarray:
    """librosa 없이 프레임 RMS로 앞뒤 무음 제거."""
    frame, hop = int(sr * 0.02), int(sr * 0.01)
    if len(y) < frame:
        return y
    n = 1 + (len(y) - frame) // hop
    rms = np.array([np.sqrt(np.mean(y[i * hop:i * hop + frame] ** 2) + 1e-12) for i in range(n)])
    thr = rms.max() * (10 ** (-top_db / 20))
    voiced = np.where(rms > thr)[0]
    if len(voiced) == 0:
        return y
    start = voiced[0] * hop
    end = min(len(y), voiced[-1] * hop + frame)
    return y[start:end]


def length_ok(y: np.ndarray, sr: int) -> bool:
    return MIN_DURATION_SEC <= len(y) / sr <= MAX_DURATION_SEC


# ------------------------------------------------------------------
# 4. 에너지 변동: 연속 게인 엔벨로프 (dB 도메인)
# ------------------------------------------------------------------
def build_envelope_db(n_samples: int, pattern: str, swing_db: float) -> np.ndarray:
    """
    길이 n_samples의 dB 게인 곡선을 만든다. 최대-최소 차이가 정확히 swing_db.
    - fade_out: 크게 시작 -> 작게 끝
    - fade_in : 작게 시작 -> 크게 끝
    - swell   : 작게 -> 크게 -> 작게
    - dip     : 크게 -> 작게 -> 크게
    코사인 곡선을 써서 미분 불연속(뚝 끊기는 느낌)이 없도록 함.
    """
    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    if pattern == "fade_out":
        shape = 0.5 * (1 + np.cos(np.pi * t))          # 1 -> 0
    elif pattern == "fade_in":
        shape = 0.5 * (1 - np.cos(np.pi * t))          # 0 -> 1
    elif pattern == "swell":
        shape = np.sin(np.pi * t)                      # 0 -> 1 -> 0
    elif pattern == "dip":
        shape = 1 - np.sin(np.pi * t)                  # 1 -> 0 -> 1
    else:
        raise ValueError(pattern)

    # 변동의 중심 위치를 살짝 흔들어 패턴이 기계적으로 똑같아지지 않게 함
    jitter = 0.08 * np.sin(2 * np.pi * (t * random.uniform(0.7, 1.6) + random.random()))
    shape = np.clip(shape + jitter, 0.0, 1.0)

    return (shape - 1.0) * swing_db                   # 최댓값 0 dB, 최솟값 -swing_db


def apply_energy_variation(y: np.ndarray, pattern: str, swing_db: float) -> np.ndarray:
    gain = 10 ** (build_envelope_db(len(y), pattern, swing_db) / 20.0)
    out = y * gain
    peak = np.max(np.abs(out))
    if peak > 0.99:                                   # 클리핑 방지
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def measure_rms_swing_db(y: np.ndarray, sr: int) -> float:
    """실제로 만들어진 클립의 구간 RMS 변동 폭(dB)을 측정해 라벨 검증용으로 기록."""
    frame, hop = int(sr * 0.05), int(sr * 0.025)
    if len(y) < frame:
        return 0.0
    n = 1 + (len(y) - frame) // hop
    rms = np.array([np.sqrt(np.mean(y[i * hop:i * hop + frame] ** 2) + 1e-12) for i in range(n)])
    voiced = rms[rms > rms.max() * 0.05]              # 무음 프레임 제외
    if len(voiced) < 2:
        return 0.0
    return float(20 * np.log10(voiced.max() / voiced.min()))


# ------------------------------------------------------------------
# 5. 샘플 1개 생성
# ------------------------------------------------------------------
def make_sample(text: str, voice: dict, pattern: str, seq: int):
    """
    파일명 규칙: {class}_{speaker_id}_{gender}_{seq:03d}.wav
      예) energy_spk009_f_001.wav
    seq 는 '화자별' 연번이다 (화자 분리 분할 시 사람이 눈으로 확인하기 쉽게).
    """
    y, sr = synthesize(text, voice["voice_id"])
    y = trim_silence(y, sr)

    if not length_ok(y, sr):
        return None, f"길이 범위 밖 ({len(y)/sr:.2f}s)"

    f0 = estimate_f0_hz(y, sr)

    swing_db = round(random.uniform(*SWING_DB_RANGE), 2)
    y = apply_energy_variation(y, pattern, swing_db)

    fname = f"{CLASS_NAME}_{voice['speaker_id']}_{voice['gender']}_{seq:03d}.wav"
    sf.write(os.path.join(AUDIO_DIR, fname), y, sr)

    return {
        # --- organized/metadata.csv 와 공통 열 (나중에 그대로 합칠 수 있게) ---
        "new_filename": fname,
        "old_filename": "",
        "class": CLASS_NAME,
        "speaker_id": voice["speaker_id"],
        "gender": voice["gender"],
        "speaker_f0_hz": f0,
        "gender_flag": "",                   # API가 성별을 알려주므로 CHECK 불필요
        "source": SOURCE_TAG,
        "voice_name": voice["name"],
        "text_script": text,
        "qc_pass": "",                       # 청취 검수 후 기입
        "cluster_raw": "",                   # 클러스터링 안 썼으므로 공란
        "is_representative": "",
        # --- energy 전용 추가 열 ---
        "batch": BATCH_NAME,
        "pattern": pattern,
        "swing_db_target": swing_db,
        "swing_db_measured": round(measure_rms_swing_db(y, sr), 2),
        "duration_sec": round(len(y) / sr, 3),
        "sample_rate": sr,
        "voice_id": voice["voice_id"],
        "base_lufs": BASE_LUFS,
        "seed": RANDOM_SEED,
    }, None


# ------------------------------------------------------------------
# 6. 실행부 (이어받기 지원)
# ------------------------------------------------------------------
META_FIELDS = [
    # organized/metadata.csv 와 동일한 앞부분 (연장 데이터와 합치기 쉽게)
    "new_filename", "old_filename", "class", "speaker_id", "gender", "speaker_f0_hz",
    "gender_flag", "source", "voice_name", "text_script", "qc_pass", "cluster_raw",
    "is_representative",
    # energy 전용 추가 열
    "batch", "pattern", "swing_db_target", "swing_db_measured", "duration_sec",
    "sample_rate", "voice_id", "base_lufs", "seed",
]


def load_done():
    if not os.path.exists(META_PATH):
        return set(), []
    with open(META_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["new_filename"] for r in rows}, rows


def _write_meta(rows):
    with open(META_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        w.writerows(rows)


def run(target_count: int = TARGET_COUNT):
    voices = pick_young_voices(NUM_VOICES_TO_USE)
    if not voices:
        raise RuntimeError("사용 가능한 청년 화자를 가져오지 못했습니다. API 키/권한을 확인하세요.")
    voices = assign_speaker_ids(voices)
    print("[화자 확정]")
    for v in voices:
        print(f"  {v['speaker_id']} / {v['gender']} / {v['name']} / {v['voice_id']}")

    done, rows = load_done()
    if done:
        print(f"[이어받기] 이미 생성된 {len(done)}개는 건너뜁니다.")

    # 화자별 연번 이어받기
    seq_by_spk = {v["speaker_id"]: 0 for v in voices}
    for r in rows:
        sid = r.get("speaker_id")
        if sid in seq_by_spk:
            seq_by_spk[sid] = max(seq_by_spk[sid], int(r["new_filename"].rsplit("_", 1)[1][:3]))

    combos = [(t, v) for t in TEXT_POOL for v in voices]
    random.shuffle(combos)

    schedule = build_pattern_schedule(target_count)
    made, idx, attempts = len(rows), 0, 0
    max_attempts = target_count * 3

    print(f"[energy] 목표 {target_count}개 / 저장 위치: {AUDIO_DIR}")
    while made < target_count and attempts < max_attempts:
        text, voice = combos[idx % len(combos)]
        pattern = schedule[made % len(schedule)]
        idx += 1
        attempts += 1

        seq = seq_by_spk[voice["speaker_id"]] + 1
        meta, skip_reason = make_sample(text, voice, pattern, seq)
        if meta is None:
            print(f"  skip ({skip_reason}) - 다른 문장으로 재시도")
            continue

        seq_by_spk[voice["speaker_id"]] = seq
        rows.append(meta)
        made += 1
        if made % 10 == 0:
            print(f"  {made}/{target_count} 완료")
            _write_meta(rows)

    _write_meta(rows)
    print(f"[energy] 완료: {made}개 / 메타데이터: {META_PATH}")
    _summary(rows)


def _summary(rows):
    if not rows:
        return
    print("\n[요약]")
    for key in ("pattern", "speaker_id", "gender"):
        counts = {}
        for r in rows:
            counts[r[key]] = counts.get(r[key], 0) + 1
        print(f"  {key}: {dict(sorted(counts.items()))}")
    measured = [float(r["swing_db_measured"]) for r in rows]
    print(f"  측정 변동 폭(dB): 평균 {np.mean(measured):.2f} / "
          f"최소 {min(measured):.2f} / 최대 {max(measured):.2f}")
    f0s = [float(r["speaker_f0_hz"]) for r in rows if float(r["speaker_f0_hz"]) > 0]
    if f0s:
        print(f"  추정 F0(Hz): 중위 {np.median(f0s):.1f} (성별 라벨은 API 값 사용)")


if __name__ == "__main__":
    run()
