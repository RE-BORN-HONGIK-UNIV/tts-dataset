"""
=============================================================================
tools/measure_energy_envelope.py  —  문장 단위 에너지 엔벨로프 지표 측정
=============================================================================

[이 파일의 역할]
  이미 만들어진 wav 파일의 "문장 전체에 걸친 음량 흐름"을 측정한다.
  음성을 합성하지 않으므로 API 크레딧이 소모되지 않는다.

  쓰임새는 두 가지다.
    1) 정상군(_pilot_normal30) 측정  -> 정상 발화가 원래 얼마나 변하는지 기준 확보
    2) 에너지변동군(energy/v4) 측정  -> 생성 결과가 정상군과 실제로 분리되는지 검증
  같은 함수로 두 집단을 재야 비교가 성립하므로 측정 코드를 한 파일로 분리했다.

[파이프라인 번호(s1~s4)가 아닌 tools/ 에 둔 이유]
  s1~s4 는 데이터를 "만드는" 단계이고 순서대로 실행된다.
  이 파일은 데이터를 "검사하는" 도구이며 어느 단계 뒤에서든 반복 실행한다.
  실행 순서에 속하지 않으므로 번호를 붙이지 않는다.

-----------------------------------------------------------------------------
[배경 - 기존 지표(swing_db)를 폐기한 이유]  ※ 발표 때 설명할 핵심
-----------------------------------------------------------------------------
  s2 파일럿의 swing_db 는 다음과 같이 정의했었다.
      50ms 프레임 RMS 중 "최대의 5% 미만"을 무음으로 제거
      -> 남은 프레임의 최대/최소 비를 dB 로 환산

  파일럿 30개 결과: 평균 24.61 dB, 최대 25.99 dB, 절반 가까이가 25.5 dB 이상.
  이 값은 우연이 아니라 정의에서 나온 천장이다.
      최소 프레임 >= 최대 x 0.05  이므로
      최대/최소 <= 1 / 0.05 = 20   ->   20 x log10(20) = 26.02 dB
  즉 거의 모든 클립이 천장에 붙어 있었다.

  원인: 50ms 는 음절보다 짧다. 이 해상도에서는 모음(큼)과 자음·단어 사이 틈(작음)의
  차이가 그대로 잡히므로, 정상 발화도 항상 20 dB 이상 출렁인다.
  swing_db 는 "문장이 점점 작아지는가"가 아니라 "음절이 얼마나 출렁이는가"를
  재고 있었고, 에너지변동 클립도 같은 천장에 붙으므로 두 클래스를 구분할 수 없다.

  교훈: 합성 사인파로 한 검증에서는 이 문제가 드러나지 않았다. 실제 음성으로
  파일럿을 돌렸기 때문에 발견했다. (본 생성 전에 파일럿을 두는 이유)

-----------------------------------------------------------------------------
[새 지표 3종]
-----------------------------------------------------------------------------
  공통 전처리
    - 25ms 창 / 10ms 간격으로 프레임 RMS 를 구해 dB 로 변환한다.
    - 유성 구간만 남긴다: 클립 최대 dB 에서 VOICED_RANGE_DB 이내인 프레임.
    - 문장 중간의 쉼(무음)은 제거한 뒤 이어 붙인다. 쉼 자체는 음량 흐름이 아니므로
      쉼을 남기면 평활 곡선이 쉼 위치에서 인위적으로 꺼진다.

  1) trend_db  (주 지표: fade_out / fade_in 판별)
       유성 프레임 뒤 25% 의 평균 dB  -  앞 25% 의 평균 dB
       음수 = 뒤로 갈수록 작아짐, 양수 = 커짐.
       ※ s2 의 trend_db 는 선형 RMS 를 평균한 뒤 dB 로 바꿨고, 여기서는 dB 값을
         평균한다. 정의가 달라 수치가 조금 다르므로 둘을 섞어 비교하지 않는다.

  2) slope_db_per_s  (보조 지표: 기울기)
       유성 프레임의 (실제 시각, dB) 에 직선을 맞춘 기울기. 단위 dB/초.
       trend_db 는 양 끝 구간만 보지만, 기울기는 전 구간을 반영한다.
       클립 길이가 달라도 "초당 몇 dB 줄어드는가"로 비교할 수 있다.

  3) env_range_db  (swell / dip 판별)
       유성 dB 곡선을 SMOOTH_SEC(0.5초) 이동평균으로 평활한 뒤 최대 - 최소.
       swell(가운데만 큼), dip(가운데만 작음)은 앞뒤가 비슷해서 trend_db 가 0 에
       가깝다. 그래서 방향과 무관한 "흐름의 폭"을 따로 잰다.
       평활 창이 음절보다 충분히 길기 때문에 음절 단위 출렁임은 평균되어 사라진다.

  비교용으로 기존 swing_db(legacy_swing_db)도 함께 계산한다.
  천장 문제를 수치로 보여주는 근거 자료다.

-----------------------------------------------------------------------------
[설정값의 근거]
-----------------------------------------------------------------------------
  FRAME 25ms / HOP 10ms
    음성 단구간 분석에서 관례적으로 쓰는 값이다. 판단 기준이 아니라 해상도 설정이다.

  VOICED_RANGE_DB = 30   (잠정값 - 학술 근거 없음)
    최대 대비 30 dB 이상 작은 프레임을 무음으로 본다. TTS 출력은 배경 잡음이 거의
    없어 이 값에 크게 민감하지 않을 것으로 예상하나, 근거가 확인되지 않은 설계값이다.

  SMOOTH_SEC = 0.5       (잠정값 - 학술 근거 없음)
    음절 2~3개 정도를 덮는 길이로 잡았다. 음절 단위 출렁임을 없애고 문장 단위 흐름만
    남기려는 의도다. 한국어 발화 속도 문헌을 확인해 근거를 보강해야 한다.

  trend 구간 25%
    s2 의 trend_db 와 같은 비율을 유지했다. 설계값이다.

-----------------------------------------------------------------------------
[입력 / 출력]
-----------------------------------------------------------------------------
  입력   측정할 배치 폴더 (기본값: data/_pilot_normal30)
           {폴더}/audio/*.wav
           {폴더}/metadata.csv   화자 이름 확인용. 없으면 파일명만으로 측정.
  출력   {폴더}/envelope_metrics.csv    클립별 지표
         콘솔                           지표별 분포 요약, 화자별 평균

[화자 제외]
  스크리닝에서 부적합으로 판정한 화자는 통계에서 뺀다. CSV 에는 남기고
  included 열을 0 으로 표시한다. (제외한 사실도 기록으로 남기기 위해)
  파일럿 판정 결과 (2026-09-24, 한국어 합성 wav 청취 기준):
      Buttaguy, Jabbaba  : 캐릭터형 발성. 면접 청년 목소리로 부적합.
      Ravi, Zoey         : 미리듣기는 영어였으나 한국어 합성은 자연스러움 -> 사용.
  -> 미리듣기 음원이 아니라 한국어로 합성한 음성으로 판정해야 한다.

[실행]
  python tools/measure_energy_envelope.py
  python tools/measure_energy_envelope.py --dir data/energy/v4
  python tools/measure_energy_envelope.py --exclude Buttaguy Jabbaba OtherName
=============================================================================
"""

import os
import csv
import glob
import argparse

import numpy as np
import soundfile as sf

# ------------------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
DEFAULT_DIR = os.path.join(DATA_ROOT, "_pilot_normal30")

# 스크리닝 결과 부적합 화자 (voice_name 기준)
EXCLUDED_VOICES = ["Buttaguy", "Jabbaba"]

FRAME_SEC = 0.025        # 프레임 창 길이
HOP_SEC = 0.010          # 프레임 간격
VOICED_RANGE_DB = 30.0   # 잠정값: 최대 dB 에서 이 범위 안이면 유성으로 본다
SMOOTH_SEC = 0.5         # 잠정값: 평활 이동평균 창 길이
TREND_PORTION = 0.25     # trend_db 에서 앞/뒤로 볼 비율

# 기존 swing_db 재현용 (비교 목적. s2 와 동일한 정의)
LEGACY_WIN_SEC = 0.05
LEGACY_HOP_SEC = 0.025
LEGACY_FLOOR_RATIO = 0.05          # 최대의 5% 미만 프레임 제거
LEGACY_CEILING_DB = 20 * np.log10(1 / LEGACY_FLOOR_RATIO)   # = 26.02 dB

EPS = 1e-10              # log(0) 방지


# ------------------------------------------------------------------
# 1. 프레임 분석
# ------------------------------------------------------------------
def frame_rms(y: np.ndarray, sr: int, win_sec: float, hop_sec: float) -> np.ndarray:
    """
    신호를 win_sec 길이 창으로 hop_sec 마다 잘라 각 창의 RMS 를 구한다.
    RMS(root mean square) = 창 안 샘플 제곱 평균의 제곱근 = 그 순간의 음량 크기.
    """
    win, hop = int(sr * win_sec), int(sr * hop_sec)
    if len(y) < win:
        return np.array([])
    n = 1 + (len(y) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]   # (프레임 수, 창 길이)
    return np.sqrt(np.mean(y[idx] ** 2, axis=1))


def voiced_db_track(y: np.ndarray, sr: int):
    """
    유성 구간의 dB 곡선과 각 프레임의 실제 시각(초)을 반환한다.

    dB 로 바꾸는 이유: 사람의 음량 지각은 로그 척도에 가깝고, 우리가 s3 에서
    부여하는 게인 엔벨로프도 dB 단위다. 같은 단위로 재야 부여값과 측정값을 비교할 수 있다.
    """
    rms = frame_rms(y, sr, FRAME_SEC, HOP_SEC)
    if len(rms) == 0:
        return np.array([]), np.array([])
    db = 20 * np.log10(rms + EPS)
    times = np.arange(len(db)) * HOP_SEC + FRAME_SEC / 2   # 프레임 중심 시각
    mask = db > (db.max() - VOICED_RANGE_DB)               # 무음(쉼) 프레임 제거
    return db[mask], times[mask]


# ------------------------------------------------------------------
# 2. 지표
# ------------------------------------------------------------------
def trend_db(db: np.ndarray) -> float:
    """뒤 25% 평균 dB - 앞 25% 평균 dB. 음수면 문장 끝으로 갈수록 작아진다."""
    if len(db) < 8:
        return float("nan")
    k = max(2, int(len(db) * TREND_PORTION))
    return float(np.mean(db[-k:]) - np.mean(db[:k]))


def slope_db_per_s(db: np.ndarray, times: np.ndarray) -> float:
    """
    (시각, dB) 에 1차 직선을 최소제곱으로 맞춘 기울기. 단위 dB/초.
    쉼을 제거했어도 x 축은 실제 시각을 쓴다. 그래야 "초당 변화량"의 의미가 유지된다.
    """
    if len(db) < 8:
        return float("nan")
    return float(np.polyfit(times, db, 1)[0])


def env_range_db(db: np.ndarray) -> float:
    """
    유성 dB 곡선을 SMOOTH_SEC 이동평균으로 평활한 뒤 최대 - 최소.
    이동평균: 각 지점에서 앞뒤 일정 구간의 평균을 취해 짧은 출렁임을 없애는 방법.
    mode="valid" 는 창이 신호 밖으로 나가는 양 끝을 버린다(가장자리 왜곡 방지).
    """
    n = int(round(SMOOTH_SEC / HOP_SEC))   # 0.5초 / 10ms = 50 프레임
    if len(db) < n + 2:
        return float("nan")                 # 클립이 평활 창보다 짧으면 측정 불가
    smooth = np.convolve(db, np.ones(n) / n, mode="valid")
    return float(smooth.max() - smooth.min())


def legacy_swing_db(y: np.ndarray, sr: int) -> float:
    """
    s2 파일럿의 기존 swing_db 를 그대로 재현한다 (비교용).
    구조적으로 LEGACY_CEILING_DB(26.02 dB)를 넘을 수 없다.
    """
    r = frame_rms(y, sr, LEGACY_WIN_SEC, LEGACY_HOP_SEC)
    if len(r) < 2:
        return float("nan")
    v = r[r > r.max() * LEGACY_FLOOR_RATIO]
    if len(v) < 2:
        return float("nan")
    return float(20 * np.log10(v.max() / v.min()))


def measure_file(path: str) -> dict:
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)            # 스테레오면 모노로
    y = y.astype(np.float64)
    db, times = voiced_db_track(y, sr)
    return {
        "duration_sec": round(len(y) / sr, 3),
        "voiced_ratio": round(len(db) / max(1, len(frame_rms(y, sr, FRAME_SEC, HOP_SEC))), 3),
        "trend_db": round(trend_db(db), 2),
        "slope_db_per_s": round(slope_db_per_s(db, times), 2),
        "env_range_db": round(env_range_db(db), 2),
        "legacy_swing_db": round(legacy_swing_db(y, sr), 2),
    }


# ------------------------------------------------------------------
# 3. 입력 읽기
# ------------------------------------------------------------------
def load_metadata(batch_dir: str) -> dict:
    """metadata.csv 가 있으면 파일명 -> 행 딕셔너리로 반환. 화자 이름·패턴 확인용."""
    path = os.path.join(batch_dir, "metadata.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r.get("new_filename", ""): r for r in csv.DictReader(f)}


# ------------------------------------------------------------------
# 4. 요약 출력
# ------------------------------------------------------------------
METRICS = [
    ("trend_db",        "dB",   "뒤 25% - 앞 25%. 음수 = 작아짐"),
    ("slope_db_per_s",  "dB/s", "직선 기울기"),
    ("env_range_db",    "dB",   "0.5초 평활 곡선의 폭"),
    ("legacy_swing_db", "dB",   f"기존 지표. 천장 {LEGACY_CEILING_DB:.2f} dB"),
]


def describe(values):
    v = np.array([x for x in values if not np.isnan(x)])
    if len(v) == 0:
        return None
    return {
        "n": len(v), "mean": v.mean(), "sd": v.std(ddof=1) if len(v) > 1 else 0.0,
        "min": v.min(), "p5": np.percentile(v, 5), "p50": np.percentile(v, 50),
        "p95": np.percentile(v, 95), "max": v.max(),
    }


def print_summary(rows, label):
    print(f"\n[{label}]  클립 {len(rows)}개 / 화자 {len({r['speaker_id'] for r in rows})}명")
    print(f"  {'지표':<16} {'n':>3} {'평균':>7} {'표준편차':>7} {'최소':>7} "
          f"{'5%':>7} {'중위':>7} {'95%':>7} {'최대':>7}")
    print("  " + "-" * 80)
    for key, unit, _ in METRICS:
        d = describe([float(r[key]) for r in rows])
        if d is None:
            continue
        print(f"  {key:<16} {d['n']:>3} {d['mean']:>7.2f} {d['sd']:>7.2f} {d['min']:>7.2f} "
              f"{d['p5']:>7.2f} {d['p50']:>7.2f} {d['p95']:>7.2f} {d['max']:>7.2f}  {unit}")
    print()
    for key, unit, desc in METRICS:
        print(f"    {key:<16} {desc}")


def print_by_speaker(rows):
    print("\n[화자별 평균]")
    print(f"  {'speaker':<8} {'voice_name':<14} {'n':>2} {'trend':>7} {'slope':>7} {'range':>7}")
    by = {}
    for r in rows:
        by.setdefault((r["speaker_id"], r["voice_name"]), []).append(r)
    for (spk, name), rs in sorted(by.items()):
        m = lambda k: np.nanmean([float(x[k]) for x in rs])
        print(f"  {spk:<8} {name:<14} {len(rs):>2} {m('trend_db'):>7.2f} "
              f"{m('slope_db_per_s'):>7.2f} {m('env_range_db'):>7.2f}")


def print_legacy_ceiling(rows):
    """기존 지표가 천장에 얼마나 붙어 있었는지 수치로 보여준다."""
    vals = [float(r["legacy_swing_db"]) for r in rows if not np.isnan(float(r["legacy_swing_db"]))]
    if not vals:
        return
    near = sum(1 for v in vals if v >= LEGACY_CEILING_DB - 1.0)
    print(f"\n[기존 swing_db 천장 확인]  천장 {LEGACY_CEILING_DB:.2f} dB 에서 1 dB 이내: "
          f"{near}/{len(vals)}개 ({100 * near / len(vals):.0f}%)")


# ------------------------------------------------------------------
# 5. 실행
# ------------------------------------------------------------------
FIELDS = ["new_filename", "speaker_id", "voice_name", "class", "pattern", "included",
          "duration_sec", "voiced_ratio", "trend_db", "slope_db_per_s", "env_range_db",
          "legacy_swing_db"]


def main():
    ap = argparse.ArgumentParser(description="문장 단위 에너지 엔벨로프 지표 측정")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="측정할 배치 폴더 (audio/ 를 포함)")
    ap.add_argument("--exclude", nargs="*", default=EXCLUDED_VOICES,
                    help="통계에서 뺄 voice_name 목록")
    args = ap.parse_args()

    batch_dir = os.path.abspath(args.dir)
    wavs = sorted(glob.glob(os.path.join(batch_dir, "audio", "*.wav")))
    if not wavs:
        raise SystemExit(f"[오류] wav 없음: {os.path.join(batch_dir, 'audio')}")

    meta = load_metadata(batch_dir)
    excluded = {x.lower() for x in (args.exclude or [])}
    print(f"[측정] {batch_dir}")
    print(f"  wav {len(wavs)}개 / 제외 화자 {sorted(args.exclude or [])}")
    print(f"  설정: 프레임 {FRAME_SEC*1000:.0f}ms/{HOP_SEC*1000:.0f}ms, "
          f"유성 범위 {VOICED_RANGE_DB:.0f}dB(잠정), 평활 {SMOOTH_SEC}s(잠정)")

    rows = []
    for p in wavs:
        fn = os.path.basename(p)
        m = meta.get(fn, {})
        parts = fn[:-4].split("_")            # {class}_{speaker_id}_{gender}_{seq}
        name = m.get("voice_name", "")
        row = {
            "new_filename": fn,
            "speaker_id": m.get("speaker_id") or (parts[1] if len(parts) > 1 else ""),
            "voice_name": name,
            "class": m.get("class") or (parts[0] if parts else ""),
            "pattern": m.get("pattern", ""),
            "included": 0 if name.lower() in excluded else 1,
        }
        row.update(measure_file(p))
        rows.append(row)

    out = os.path.join(batch_dir, "envelope_metrics.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  저장: {out}")

    inc = [r for r in rows if r["included"] == 1]
    exc = [r for r in rows if r["included"] == 0]
    if exc:
        print(f"  통계 제외: {len(exc)}개 ({', '.join(sorted({r['voice_name'] for r in exc}))})")

    print("\n" + "=" * 84)
    print(" 문장 단위 에너지 엔벨로프 분포")
    print("=" * 84)
    print_summary(inc, "사용 화자")

    # 에너지변동 배치처럼 pattern 이 기록돼 있으면 패턴별로도 요약한다
    patterns = sorted({r["pattern"] for r in inc if r["pattern"] and r["pattern"] != "none"})
    for pat in patterns:
        print_summary([r for r in inc if r["pattern"] == pat], f"pattern = {pat}")

    print_by_speaker(inc)
    print_legacy_ceiling(rows)

    print("\n" + "-" * 84)
    print(" 해석 방법")
    print("  - 정상군 trend_db 의 5% 값보다 더 작아지는 클립 -> fade_out 후보 영역")
    print("  - 정상군 trend_db 의 95% 값보다 더 커지는 클립 -> fade_in 후보 영역")
    print("  - 정상군 env_range_db 의 95% 값을 넘는 클립     -> swell / dip 후보 영역")
    print("  - 위 값은 분포 요약일 뿐 임계값이 아니다. 표본 수와 화자 수를 함께 보고 결정한다.")
    print("-" * 84)


if __name__ == "__main__":
    main()
