"""
=============================================================================
s2_generate_normal_pilot.py  —  정상(normal) 음성 30개 파일럿 + 변동 폭 실측
=============================================================================

[이 파일의 역할]
  화자 10명 x 3개 = 정상 발화 30개를 생성하고, 각 클립의 "문장 내 에너지 변동
  폭"을 실측해 분포를 출력한다. 학습용 데이터가 아니라 측정용 데이터다.

[왜 이 단계가 필요한가 - 핵심]
  우리가 검출하려는 '에너지 변동'은 면접 중 자신감이 떨어져 목소리가 점점
  작아지거나(fade out), 작게 시작했다가 커지는(fade in) 현상이다.
  그런데 문제가 있다. 정상 발화에도 음량이 줄어드는 경향이 원래 존재한다.
  발화가 진행되며 F0 와 강도가 함께 감소하는 현상을 declination 이라 하며,
  언어 보편적 경향으로 보고된다. 특히 평서문은 종결부에서 dB SPL 이 크게
  떨어지는 것이 특징으로 기술된다.
      -> 즉 "줄어드는가"가 아니라 "비정상적으로 크게 줄어드는가"가 기준이어야 한다.
  따라서 정상군의 변동 폭 분포를 먼저 실측하지 않으면 임계값을 정할 수 없다.

[왜 정상군도 TTS 로 만드는가]
  정상군을 실제 녹음으로 구하면, 마이크 특성·배경 잡음·대역폭이 클래스와
  완벽하게 상관되어 모델이 "합성음이냐 실제 녹음이냐"를 학습해 버린다.
  정상군과 에너지변동군을 같은 화자·같은 문장·같은 target_lufs 로 생성하면
  두 클래스의 차이가 음량 변동 하나로 통제된다.
  (합성 데이터가 의도치 않은 편향을 새로 들여올 수 있다는 지적은 아래 참고 자료)

[정상군에 인위적 변동을 주입하지 않는 이유]
  Typecast 는 사람 음성으로 학습된 모델이라 출력에 declination 이 이미 들어 있다.
  인위적으로 넣으면 이중으로 적용되어 오히려 왜곡된다. 그대로 쓰고 측정만 한다.

[측정하는 두 지표]
  swing_db  : 클립 내 구간 RMS 의 최대/최소 비를 dB 로 환산한 값.
              무음 프레임(최대의 5% 미만)은 제외한다.
              -> "이 클립이 얼마나 크게 출렁였는가"
  trend_db  : 문장 뒷부분 평균 - 앞부분 평균 (dB).
              음수면 작아짐(declination / fade out), 양수면 커짐(fade in).
              -> "어느 방향으로 흘렀는가"
  두 지표를 나눈 이유: swing 만 보면 방향을 알 수 없고, trend 만 보면 크기를
  알 수 없다. 우리 타깃은 "큰 폭으로, 한 방향으로" 흐르는 경우다.

[미리듣기 음원을 같이 내려받는 이유]
  API 에 언어 정보가 없으므로(s1 참고) 외국인 억양 화자는 귀로 걸러야 한다.
  v3 상세 조회의 preview_url 을 받아 _previews 폴더에 저장한다.
  10명이면 30초면 확인이 끝난다.

[왜 폴더 이름이 밑줄로 시작하는가]
  _pilot_normal30 은 학습에 쓰지 않는 실험 데이터다. 04 통합 스크립트가
  밑줄로 시작하는 폴더를 건너뛰므로, 마스터 메타데이터에 섞이지 않는다.
  그래도 CSV 는 남긴다. 임계값을 정한 근거 자료이므로 논문·발표에 필요하다.

[입력]  data/voice_candidates.csv  (s1 의 출력)
[출력]  data/_pilot_normal30/audio/normal_spkP01_f_001.wav ...
        data/_pilot_normal30/metadata.csv
        data/_pilot_normal30/_previews/    화자별 미리듣기 음원
        콘솔에 swing_db / trend_db 분포 히스토그램

[화자 ID 규칙]
  파일럿 화자는 spkP01 처럼 P 를 붙인다. 본 생성의 spk001 번호와 겹치지 않게 해서
  나중에 데이터가 섞여도 출처를 구분할 수 있다.

[참고 자료]
  발화 중 F0/강도 declination (Lieberman 인용)
    https://repository.uantwerpen.be/docman/irua/0b0ce0/154885.pdf
  평서문 종결부의 dB SPL 급감 보고
    https://era.library.ualberta.ca/items/a14f3137-5fe8-46bb-a6a0-dbf7d48c1a8f/download/61497a8d-33b7-477d-93a2-15dfabac3338
  발화 말/비말 강도 비 1.33~5.76 dB 측정 사례
    https://pmc.ncbi.nlm.nih.gov/articles/PMC3212410/
  합성 데이터가 편향을 도입·증폭할 수 있다는 지적
    https://direct.mit.edu/coli/article/51/1/191/124625/Evaluating-Synthetic-Data-Generation-from-User

[실행]
  pip install typecast-python requests python-dotenv soundfile numpy
  python s2_generate_normal_pilot.py
=============================================================================
"""

import os
import io
import csv
import sys
import time
import random
import datetime

import numpy as np
import requests
import soundfile as sf
from dotenv import load_dotenv
from typecast import Typecast
from typecast.models import TTSRequest, Output, LanguageCode

load_dotenv()

# ------------------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------------
SCRIPT_VERSION = "pilot_v4"       # metadata 에 기록됨

# 데이터 폴더 위치
#   코드 파일(src/)의 한 칸 위 = repo 폴더, 그 안의 data/ 를 쓴다.
#   C:\... 같은 절대경로를 쓰지 않는 이유: repo 를 다른 곳(예: OneDrive 밖)으로
#   옮겨도 코드를 고칠 필요가 없고, 어디서 실행하든 같은 폴더를 가리킨다.
#   data/ 는 .gitignore 에 들어 있어 GitHub 에 올라가지 않는다(음성 용량 문제).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
CANDIDATES_CSV = os.path.join(DATA_ROOT, "voice_candidates.csv")

OUT_DIR = os.path.join(DATA_ROOT, "_pilot_normal30")   # 밑줄 = 마스터 제외
AUDIO_DIR = os.path.join(OUT_DIR, "audio")
PREVIEW_DIR = os.path.join(OUT_DIR, "_previews")
META_PATH = os.path.join(OUT_DIR, "metadata.csv")

N_SPEAKERS = 10                   # 남 5 / 여 5
CLIPS_PER_SPEAKER = 3             # 총 30개
CLASS_NAME = "normal"
SOURCE_TAG = "tts"

TTS_MODEL = "ssfm-v30"
BASE_LUFS = -18.0                 # 본 생성과 동일하게 맞춰야 비교가 성립한다
MIN_DURATION_SEC = 3.0
MAX_DURATION_SEC = 5.0

# check_voices.py 결과에서 추가로 제외할 태그 (협의 결과: Anime/Game 제외)
EXTRA_EXCLUDE = ["anime", "game", "rapper"]

SILENCE_TRIM_DB = 40.0
MAX_RETRY = 3
RETRY_SLEEP_SEC = 2.0
RANDOM_SEED = 20260922

API_KEY = os.getenv("TYPECAST_API_KEY")
VOICE_DETAIL_URL = "https://api.typecast.ai/v3/voices/{voice_id}"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

client = Typecast()

# 화자 10명 x 3개 = 30개. 문장이 겹치지 않도록 30개 이상 준비.
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
]


# ------------------------------------------------------------------
# 1. 후보 화자 읽기 + 10명 선정
# ------------------------------------------------------------------
def load_candidates():
    if not os.path.exists(CANDIDATES_CSV):
        sys.exit(f"[오류] {CANDIDATES_CSV} 가 없습니다. check_voices.py 를 먼저 실행하세요.")

    with open(CANDIDATES_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    def ok(r):
        if (r.get("excluded_by") or "").strip():          # 이미 제외된 화자
            return False
        tags = (r.get("use_cases") or "").lower()
        return not any(kw in tags for kw in EXTRA_EXCLUDE)

    kept = [r for r in rows if ok(r)]
    print(f"[후보] 전체 {len(rows)}명 -> 조건 통과 {len(kept)}명 "
          f"(제외 태그: {EXTRA_EXCLUDE})")
    return kept


def pick_speakers(cands, n=N_SPEAKERS):
    males = [c for c in cands if c.get("gender") == "male"]
    females = [c for c in cands if c.get("gender") == "female"]
    random.shuffle(males)
    random.shuffle(females)

    half = n // 2
    picked = males[:half] + females[:n - half]
    if len(picked) < n:
        sys.exit(f"[오류] 화자가 부족합니다 (남 {len(males)} / 여 {len(females)})")

    random.shuffle(picked)
    out = []
    for i, c in enumerate(picked, start=1):
        out.append({
            "speaker_id": f"spkP{i:02d}",          # P = pilot. 본 생성 번호와 겹치지 않게
            "voice_id": c["voice_id"],
            "name": c.get("voice_name", ""),
            "gender": "f" if c.get("gender") == "female" else "m",
            "use_cases": c.get("use_cases", ""),
        })
    return out


def download_previews(speakers):
    """v3 상세 조회로 preview_url 을 받아 미리듣기 음원을 저장. 억양 확인용."""
    if not API_KEY:
        print("[경고] TYPECAST_API_KEY 없음 -> 미리듣기 건너뜀")
        return
    print(f"\n[미리듣기] {len(speakers)}명 음원 다운로드 -> {PREVIEW_DIR}")
    for s in speakers:
        try:
            res = requests.get(VOICE_DETAIL_URL.format(voice_id=s["voice_id"]),
                               headers={"X-API-KEY": API_KEY}, timeout=20)
            res.raise_for_status()
            d = res.json()
            url = d.get("preview_url")
            vn = d.get("voice_name") or {}
            if isinstance(vn, dict):
                s["name_kor"] = vn.get("kor", "")
                s["name_eng"] = vn.get("eng", s["name"])
            if not url:
                print(f"  {s['speaker_id']} {s['name']}: preview_url 없음")
                continue
            ext = os.path.splitext(url.split("?")[0])[1] or ".mp3"
            path = os.path.join(PREVIEW_DIR,
                                f"{s['speaker_id']}_{s['gender']}_{s['name']}{ext}")
            audio = requests.get(url, timeout=30)
            audio.raise_for_status()
            with open(path, "wb") as f:
                f.write(audio.content)
            print(f"  {s['speaker_id']} {s['name']:<18} {s.get('name_kor','')}")
        except Exception as e:
            print(f"  {s['speaker_id']} {s['name']}: 실패 ({e})")


# ------------------------------------------------------------------
# 2. TTS 합성 (변동 주입 없음. 원본 그대로)
# ------------------------------------------------------------------
def synthesize(text: str, voice_id: str):
    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            res = client.text_to_speech(TTSRequest(
                text=text,
                model=TTS_MODEL,
                voice_id=voice_id,
                language=LanguageCode.KOR,
                output=Output(
                    target_lufs=BASE_LUFS,
                    audio_pitch=0,
                    audio_tempo=1.0,
                    audio_format="wav",
                ),
            ))
            y, sr = sf.read(io.BytesIO(res.audio_data))
            if y.ndim > 1:
                y = y.mean(axis=1)
            return y.astype(np.float32), sr
        except Exception as e:
            last_err = e
            print(f"    [재시도 {attempt}/{MAX_RETRY}] {type(e).__name__}: {e}")
            time.sleep(RETRY_SLEEP_SEC * attempt)
    raise RuntimeError(f"TTS 실패: {last_err}")


# ------------------------------------------------------------------
# 3. 측정 유틸
# ------------------------------------------------------------------
def trim_silence(y, sr, top_db=SILENCE_TRIM_DB):
    frame, hop = int(sr * 0.02), int(sr * 0.01)
    if len(y) < frame:
        return y
    n = 1 + (len(y) - frame) // hop
    rms = np.array([np.sqrt(np.mean(y[i*hop:i*hop+frame] ** 2) + 1e-12) for i in range(n)])
    thr = rms.max() * (10 ** (-top_db / 20))
    voiced = np.where(rms > thr)[0]
    if len(voiced) == 0:
        return y
    return y[voiced[0]*hop: min(len(y), voiced[-1]*hop + frame)]


def rms_frames(y, sr, win=0.05, hop=0.025):
    frame, hp = int(sr*win), int(sr*hop)
    if len(y) < frame:
        return np.array([])
    n = 1 + (len(y) - frame) // hp
    return np.array([np.sqrt(np.mean(y[i*hp:i*hp+frame] ** 2) + 1e-12) for i in range(n)])


def measure_swing_db(y, sr):
    """
    클립 내 구간 RMS 변동 폭(dB). 본 생성 스크립트와 동일한 정의를 써야
    정상군/energy 비교가 성립한다.
    무음 프레임(최대의 5% 미만)은 제외한다.
    """
    r = rms_frames(y, sr)
    if len(r) < 2:
        return 0.0
    voiced = r[r > r.max() * 0.05]
    if len(voiced) < 2:
        return 0.0
    return float(20 * np.log10(voiced.max() / voiced.min()))


def measure_trend_db(y, sr):
    """
    문장 앞부분 대비 뒷부분의 에너지 차이(dB).
    양수면 커짐(fade in 경향), 음수면 작아짐(fade out / declination).
    정상 발화에도 자연스러운 declination 이 있으므로 그 크기를 본다.
    """
    r = rms_frames(y, sr)
    r = r[r > r.max() * 0.05] if len(r) else r
    if len(r) < 6:
        return 0.0
    k = max(2, len(r) // 4)
    head = np.mean(r[:k])
    tail = np.mean(r[-k:])
    return float(20 * np.log10(tail / head))


def histogram(values, lo=None, hi=None, bins=10, width=34):
    if not values:
        return
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    if hi <= lo:
        hi = lo + 1
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    mx = max(counts) if len(counts) else 1
    for i in range(bins):
        n = counts[i]
        bar = "#" * int(round(width * n / mx)) if mx else ""
        print(f"   {edges[i]:>6.1f} ~ {edges[i+1]:>6.1f} dB | {n:>3}개 {bar}")


# ------------------------------------------------------------------
# 4. 실행
# ------------------------------------------------------------------
META_FIELDS = [
    "new_filename", "old_filename", "class", "speaker_id", "gender", "speaker_f0_hz",
    "gender_flag", "source", "voice_name", "text_script", "qc_pass", "cluster_raw",
    "is_representative",
    "batch", "pattern", "swing_db_target", "swing_db_measured", "trend_db",
    "duration_sec", "sample_rate", "voice_id", "base_lufs", "seed",
    "use_cases", "lang_flag", "script_version", "generated_at",
]


def main():
    speakers = pick_speakers(load_candidates())
    print("\n[선정 화자]")
    for s in speakers:
        print(f"  {s['speaker_id']}  {s['gender']}  {s['name']:<18} {s['use_cases']}")

    download_previews(speakers)

    texts = TEXT_POOL[:]
    random.shuffle(texts)
    ti = 0
    rows = []
    now = datetime.datetime.now().isoformat(timespec="seconds")

    total = len(speakers) * CLIPS_PER_SPEAKER
    print(f"\n[생성] 정상음성 {total}개 -> {AUDIO_DIR}")
    for s in speakers:
        seq = 0
        tries = 0
        while seq < CLIPS_PER_SPEAKER and tries < CLIPS_PER_SPEAKER * 4:
            text = texts[ti % len(texts)]
            ti += 1
            tries += 1

            y, sr = synthesize(text, s["voice_id"])
            y = trim_silence(y, sr)
            dur = len(y) / sr
            if not (MIN_DURATION_SEC <= dur <= MAX_DURATION_SEC):
                print(f"  skip {s['speaker_id']} ({dur:.2f}s 범위 밖)")
                continue

            seq += 1
            fname = f"{CLASS_NAME}_{s['speaker_id']}_{s['gender']}_{seq:03d}.wav"
            sf.write(os.path.join(AUDIO_DIR, fname), y, sr)

            swing = round(measure_swing_db(y, sr), 2)
            trend = round(measure_trend_db(y, sr), 2)
            rows.append({
                "new_filename": fname, "old_filename": "", "class": CLASS_NAME,
                "speaker_id": s["speaker_id"], "gender": s["gender"],
                "speaker_f0_hz": "", "gender_flag": "", "source": SOURCE_TAG,
                "voice_name": s["name"], "text_script": text, "qc_pass": "",
                "cluster_raw": "", "is_representative": "",
                "batch": "_pilot_normal30", "pattern": "none",
                "swing_db_target": "", "swing_db_measured": swing, "trend_db": trend,
                "duration_sec": round(dur, 3), "sample_rate": sr,
                "voice_id": s["voice_id"], "base_lufs": BASE_LUFS, "seed": RANDOM_SEED,
                "use_cases": s["use_cases"], "lang_flag": "",
                "script_version": SCRIPT_VERSION, "generated_at": now,
            })
            print(f"  {fname:<30} swing={swing:>6.2f}dB  trend={trend:>+6.2f}dB  {dur:.2f}s")

    with open(META_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[저장] {META_PATH}  ({len(rows)}행)")

    report(rows)


def report(rows):
    if not rows:
        return
    swings = [float(r["swing_db_measured"]) for r in rows]
    trends = [float(r["trend_db"]) for r in rows]

    print("\n" + "=" * 70)
    print(" 정상 발화의 문장 내 에너지 변동 폭 (energy 임계값 근거 자료)")
    print("=" * 70)
    print(f"\n swing_db  n={len(swings)}  "
          f"평균 {np.mean(swings):.2f}  중위 {np.median(swings):.2f}  "
          f"표준편차 {np.std(swings):.2f}")
    print(f"           최소 {min(swings):.2f}  최대 {max(swings):.2f}")
    for p in (50, 75, 90, 95, 99):
        print(f"           {p}백분위 {np.percentile(swings, p):.2f} dB")
    print("\n [분포]")
    histogram(swings)

    print(f"\n trend_db (문장 뒷부분 - 앞부분, 음수면 작아짐)")
    print(f"           평균 {np.mean(trends):+.2f}  중위 {np.median(trends):+.2f}  "
          f"최소 {min(trends):+.2f}  최대 {max(trends):+.2f}")
    n_down = sum(1 for t in trends if t < 0)
    print(f"           작아지는 클립 {n_down}/{len(trends)}개 "
          f"({100*n_down/len(trends):.0f}%) <- declination 확인")
    print("\n [분포]")
    histogram(trends)

    print("\n [화자별 평균 swing_db]")
    by_spk = {}
    for r in rows:
        by_spk.setdefault(r["speaker_id"], []).append(float(r["swing_db_measured"]))
    for spk in sorted(by_spk):
        v = by_spk[spk]
        print(f"   {spk}  평균 {np.mean(v):>6.2f} dB  (n={len(v)})")

    p95 = np.percentile(swings, 95)
    print("\n" + "-" * 70)
    print(f" 해석 참고: 정상군 95백분위가 {p95:.1f} dB 이므로,")
    print(f"           energy 클래스는 이보다 충분히 큰 값에서 시작해야 두 분포가 겹치지 않음.")
    print(f"           (현재 energy 잠정 설정 10~20 dB 를 이 수치와 비교해 조정)")
    print("-" * 70)
    print("\n 다음 할 일")
    print("  1) _previews 폴더의 음원을 들어 외국인 억양 화자를 찾고,")
    print("     metadata.csv 의 lang_flag 열에 ko / foreign 을 기입")
    print("  2) audio 폴더의 wav 30개를 들어 정상 발화로 들리는지 확인")
    print("  3) 위 분포 수치를 공유해 주시면 energy 변동 폭을 확정")


if __name__ == "__main__":
    main()
