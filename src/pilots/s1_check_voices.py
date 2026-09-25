"""
=============================================================================
s1_check_voices.py  —  Typecast 화자 조회 및 조건 필터링 (읽기 전용)
=============================================================================

[이 파일의 역할]
  학습 데이터 생성에 쓸 "화자 풀"을 확정한다.
  음성을 합성하지 않고 GET /v2/voices 만 호출하므로 API 크레딧이 소모되지 않는다.

[왜 필요한가]
  화자 수는 데이터 개수보다 중요하다. 화자 5명으로 1,200개를 만들면 모델이
  그 5명의 목소리를 외워버려서(speaker overfitting) 실제 사용자에게는 동작하지
  않는다. 그래서 생성 전에 "조건에 맞는 화자가 몇 명 확보되는지"를 먼저 센다.

[우리 조건과 그 이유]
  - age = young_adult : 공식 문서상 20~35세. 서비스 대상인 고립청년 연령대와 일치.
  - model = ssfm-v30  : 운율 품질이 개선된 최신 모델. whisper 등 감정 프리셋 지원.
  - voice_type = original : custom 은 계정에서 복제한 음성이라 재현성이 없어 제외.
  - use_cases 에서 Rapper 제외 : 랩 발성은 면접 발화와 운율 특성이 전혀 다르다.
  - use_cases 에서 Anime / Game 제외 : 캐릭터성 과장 발성 위험.

[중요 - 공식 문서와 실제 API 응답이 다르다]
  문서에는 use_cases 가 Audiobook, E-learning, News, Podcast, Voicemail, Ads,
  Tiktok/Reels 로 적혀 있으나, 실제 응답은 아래처럼 확장된 이름을 반환한다.
      Audiobook      -> Audiobook/Storytelling
      E-learning     -> E-learning/Explainer
      Podcast        -> Radio/Podcast
      Tiktok/Reels   -> TikTok/Reels/Shorts
      Ads            -> Ads/Promotion
      Voicemail      -> Voicemail/Voice Assistant
      News           -> News Reporter
  Rapper, Anime, Game, Conversational, Documentary, Announcer 는 문서와 동일하다.
  그래서 이 스크립트는 태그를 소문자 부분일치로 비교한다(정확히 일치가 아님).

[알려진 한계 - 언어 정보가 없다]
  Typecast API 응답에는 language / locale / nationality 필드가 전혀 없다.
  (List Voices, Get Voice Details 양쪽 모두 확인)
  따라서 일본어·중국어 화자가 같은 목록에 섞여 있어도 API 로는 구분할 수 없다.
  실제 조회 결과에도 Yui Sato, Kenta Yoshida, Hao Ran 같은 화자가 포함됐다.
  이런 화자에게 language=KOR 로 한국어를 시키면 외국인 억양이 나오므로,
  다음 단계(02)에서 미리듣기 음원을 듣고 걸러낸다.

[입력]  없음 (.env 의 TYPECAST_API_KEY 만 필요)
[출력]  data/voice_candidates.csv
          excluded_by 열이 빈 행 = 사용 가능한 화자
          이후 모든 생성 스크립트가 이 CSV 를 읽는다. API 를 매번 다시 조회하지
          않으므로 "어떤 화자 풀로 만든 데이터인가"가 기록으로 남는다.

[참고 자료]
  Typecast List Voices 문서
    https://typecast.ai/docs/api-reference/voices/list-voices
  Typecast Get Voice Details 문서 (preview_url, voice_name.kor 제공)
    https://typecast.ai/docs/api-reference/voices/get-voice-details

[실행]
  pip install requests python-dotenv
  python s1_check_voices.py
=============================================================================
"""

import os
import csv
import sys
from collections import Counter, defaultdict

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TYPECAST_API_KEY")
BASE_URL = "https://api.typecast.ai/v2/voices"

TARGET_AGE = "young_adult"          # 20~35세 (고립청년 타깃과 일치)
TARGET_MODEL = "ssfm-v30"
ONLY_ORIGINAL = True                # 복제 음성(custom) 제외

# 1차 제외 태그. 소문자 부분일치로 비교한다.
EXCLUDE_USE_CASES = ["rapper"]

# 2차 후보(과장된 캐릭터 발성 우려). 인원이 충분하면 이것도 빼는 걸 검토.
CANDIDATE_EXTRA_EXCLUDE = ["anime", "game"]

# 데이터 폴더 위치
#   코드 파일(src/)의 한 칸 위 = repo 폴더, 그 안의 data/ 를 쓴다.
#   C:\... 같은 절대경로를 쓰지 않는 이유: repo 를 다른 곳(예: OneDrive 밖)으로
#   옮겨도 코드를 고칠 필요가 없고, 어디서 실행하든 같은 폴더를 가리킨다.
#   data/ 는 .gitignore 에 들어 있어 GitHub 에 올라가지 않는다(음성 용량 문제).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
# 결과 명단. s2·s3 가 이 파일을 읽어 화자를 뽑는다.
OUT_CSV = os.path.join(DATA_ROOT, "voice_candidates.csv")

# 공식 문서 기준 use_cases 전체 13종
ALL_USE_CASES = [
    "Announcer", "Anime", "Audiobook", "Conversational", "Documentary",
    "E-learning", "Rapper", "Game", "Tiktok/Reels", "News", "Podcast",
    "Voicemail", "Ads",
]


def fetch_voices(age=None, model=None):
    """GET /v2/voices. age/model 을 주면 서버에서 필터링해서 받는다."""
    if not API_KEY:
        sys.exit("[오류] TYPECAST_API_KEY 가 없습니다. .env 파일을 확인하세요.")

    params = {}
    if age:
        params["age"] = age
    if model:
        params["model"] = model

    res = requests.get(BASE_URL, headers={"X-API-KEY": API_KEY}, params=params, timeout=30)
    if res.status_code == 401:
        sys.exit("[오류] 401 Unauthorized. API 키가 잘못되었거나 만료되었습니다.")
    res.raise_for_status()
    return res.json()


def norm_use_cases(v):
    return [str(t).strip() for t in (v.get("use_cases") or [])]


def has_any(tags, keywords):
    low = [t.lower() for t in tags]
    return any(kw in t for t in low for kw in keywords)


def supports_model(v, model):
    return any(m.get("version") == model for m in (v.get("models") or []))


def emotions_for(v, model):
    for m in (v.get("models") or []):
        if m.get("version") == model:
            return m.get("emotions") or []
    return []


def bar(n, total, width=30):
    if total <= 0:
        return ""
    filled = int(round(width * n / total))
    return "#" * filled + "." * (width - filled)


def main():
    print("=" * 74)
    print(f" Typecast 화자 조회  |  age={TARGET_AGE}  model={TARGET_MODEL}")
    print("=" * 74)

    # 참고용: 전체 인원도 같이 받아 청년 비율을 본다
    try:
        all_voices = fetch_voices()
        print(f"\n계정에서 조회 가능한 전체 화자: {len(all_voices)}명")
        age_dist = Counter((v.get("age") or "unknown") for v in all_voices)
        for a, n in age_dist.most_common():
            print(f"   {a:<14} {n:>5}명  {bar(n, len(all_voices))}")
    except Exception as e:
        print(f"[경고] 전체 조회 실패({e}). 청년 조회만 진행합니다.")

    voices = fetch_voices(age=TARGET_AGE, model=TARGET_MODEL)
    print(f"\n── 1단계: age={TARGET_AGE}, model={TARGET_MODEL} → {len(voices)}명")

    # voice_type 필터
    if ONLY_ORIGINAL:
        before = len(voices)
        customs = [v for v in voices if v.get("voice_type") != "original"]
        voices = [v for v in voices if v.get("voice_type") == "original"]
        print(f"── 2단계: custom(복제 음성) {before - len(voices)}명 제외 → {len(voices)}명")
        if customs:
            print("     제외된 custom: "
                  + ", ".join(v.get("voice_name", "?") for v in customs[:10]))

    # model 지원 재확인 (서버 필터를 신뢰하되 방어적으로)
    voices = [v for v in voices if supports_model(v, TARGET_MODEL)]

    # use_cases 분포
    print(f"\n── use_cases 태그 분포 (청년 {len(voices)}명 기준)")
    tag_count = Counter()
    no_tag = 0
    for v in voices:
        tags = norm_use_cases(v)
        if not tags:
            no_tag += 1
        for t in tags:
            tag_count[t] += 1
    for t in ALL_USE_CASES:
        n = tag_count.get(t, 0)
        mark = "  <- 제외 대상" if t.lower() in EXCLUDE_USE_CASES else (
            "  <- 제외 검토" if t.lower() in CANDIDATE_EXTRA_EXCLUDE else "")
        print(f"   {t:<14} {n:>5}명  {bar(n, len(voices))}{mark}")
    other = [t for t in tag_count if t not in ALL_USE_CASES]
    for t in other:
        print(f"   {t:<14} {tag_count[t]:>5}명   (문서에 없는 태그)")
    if no_tag:
        print(f"   (태그 없음)    {no_tag:>5}명")

    # 1차 제외 적용
    excluded = [v for v in voices if has_any(norm_use_cases(v), EXCLUDE_USE_CASES)]
    kept = [v for v in voices if not has_any(norm_use_cases(v), EXCLUDE_USE_CASES)]

    print(f"\n── 3단계: {EXCLUDE_USE_CASES} 태그 {len(excluded)}명 제외 → 사용 가능 {len(kept)}명")
    for v in excluded[:20]:
        print(f"     제외  {v.get('voice_name','?'):<20} {use_cases_str(v)}")
    if len(excluded) > 20:
        print(f"     ... 외 {len(excluded) - 20}명")

    # 성별 분포
    g = Counter((v.get("gender") or "unknown") for v in kept)
    print(f"\n── 성별 분포: " + ", ".join(f"{k}={n}명" for k, n in g.most_common()))
    balanced = min(g.get("male", 0), g.get("female", 0)) * 2
    print(f"   남녀 1:1 로 맞출 경우 최대 {balanced}명 사용 가능")

    # 2차 제외를 추가하면 몇 명이 되는지 미리보기 (실제 적용은 안 함)
    extra_kept = [v for v in kept if not has_any(norm_use_cases(v), CANDIDATE_EXTRA_EXCLUDE)]
    eg = Counter((v.get("gender") or "unknown") for v in extra_kept)
    print(f"\n── (참고) {CANDIDATE_EXTRA_EXCLUDE} 까지 제외하면 → {len(extra_kept)}명 "
          f"(" + ", ".join(f"{k}={n}" for k, n in eg.most_common()) + ")")
    print("   인원이 넉넉하면 이쪽이 면접 음성에 더 적합합니다.")

    # whisper 감정 지원 여부
    whisper_ok = sum(1 for v in kept if "whisper" in emotions_for(v, TARGET_MODEL))
    print(f"\n── whisper 감정 지원: {whisper_ok}/{len(kept)}명")
    emo_dist = Counter()
    for v in kept:
        for e in emotions_for(v, TARGET_MODEL):
            emo_dist[e] += 1
    print("   감정 분포: " + ", ".join(f"{e}={n}" for e, n in emo_dist.most_common()))

    # 명단 저장
    fields = ["voice_id", "voice_name", "gender", "age", "voice_type",
              "use_cases", "emotions", "excluded_by"]
    os.makedirs(DATA_ROOT, exist_ok=True)   # 처음 실행이면 data/ 폴더 생성
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in kept + excluded:
            tags = norm_use_cases(v)
            reason = ""
            if has_any(tags, EXCLUDE_USE_CASES):
                reason = "primary_exclude"
            elif has_any(tags, CANDIDATE_EXTRA_EXCLUDE):
                reason = "extra_exclude_candidate"
            w.writerow({
                "voice_id": v.get("voice_id", ""),
                "voice_name": v.get("voice_name", ""),
                "gender": v.get("gender", ""),
                "age": v.get("age", ""),
                "voice_type": v.get("voice_type", ""),
                "use_cases": "|".join(tags),
                "emotions": "|".join(emotions_for(v, TARGET_MODEL)),
                "excluded_by": reason,
            })

    print(f"\n[저장] {os.path.abspath(OUT_CSV)}")
    print("   excluded_by 열이 빈 행이 바로 사용 가능한 화자입니다.")
    print("   이 CSV 를 생성 스크립트가 읽게 하면 조회를 매번 반복하지 않아도 됩니다.")

    # 앞쪽 몇 명 미리보기
    print(f"\n── 사용 가능 화자 미리보기 (앞 15명)")
    print(f"   {'voice_name':<20}{'gender':<9}{'use_cases'}")
    for v in kept[:15]:
        print(f"   {v.get('voice_name','?'):<20}{str(v.get('gender','')):<9}"
              f"{'|'.join(norm_use_cases(v))}")

 


def use_cases_str(v):
    return "|".join(norm_use_cases(v))


if __name__ == "__main__":
    main()
