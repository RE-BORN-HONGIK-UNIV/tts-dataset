"""
=============================================================================
s4_build_master_metadata.py  —  배치별 메타데이터를 마스터 한 장으로 통합
=============================================================================

[이 파일의 역할]
  배치 폴더마다 흩어진 metadata.csv 를 모아, 클래스별로 정렬된 마스터 CSV
  한 장을 만든다. 클래스별 분리본과 데이터 균형 점검표도 함께 출력한다.

  정렬 순서:  class(normal -> prolong -> energy -> tremor)
              -> speaker_id -> 화자별 연번
  결과적으로 연장은 연장끼리 쭉, 그 다음 에너지변동이 쭉 이어진다.

[왜 생성 스크립트가 마스터에 직접 쓰지 않는가]
  생성 중 중단되거나 여러 클래스를 동시에 작업하면 한 파일을 여러 쪽에서
  건드려 깨질 수 있다. 또 append 방식이면 "연장 83개 뒤에 떨림이 쌓이는"
  뒤섞인 순서가 된다.
  그래서 배치 CSV 는 작업 로그로만 두고, 마스터는 '항상 다시 만드는' 파일로
  둔다. 마스터를 직접 손으로 고치지 말고, 수정은 배치 CSV 에서 하고 이 스크립트를
  다시 실행한다.

[밑줄 규칙]
  이름이 밑줄(_)로 시작하는 폴더는 통합에서 제외한다.
      _pilot_normal30        측정용 파일럿 (학습에 쓰지 않음)
      _test_batch1/2/3       과거 코드 테스트 결과물
      _archive_...           보관만 하는 구 데이터
  폴더 이름만 보고 "학습용인가 실험용인가"를 구분할 수 있게 하는 장치다.

[교차표를 출력하는 이유 - 발표 때 설명할 포인트]
  클래스 x 화자 교차표를 보면 한 화자가 한 클래스에만 등장하는지 알 수 있다.
  예를 들어 연장은 spk001~003, 에너지는 spk005~006 만 있으면, 모델이 음향
  패턴이 아니라 "누구 목소리인가"를 외워서 분류할 수 있다(speaker leakage).
  같은 화자로 여러 클래스를 만들어야 이 문제가 사라진다.
  또한 학습/검증/테스트 분할은 반드시 화자 단위로 해야 한다. 같은 화자가
  분할을 넘나들면 성능이 과대평가된다.

[입력]  data/ 아래의 모든 metadata.csv (밑줄 폴더 제외)
[출력]  data/master_metadata.csv          전체 한 장, 클래스별 정렬
        data/by_class/{class}.csv         클래스별 분리본 (눈으로 확인할 때 편함)
        콘솔에 클래스별 개수 + 클래스 x 화자 교차표 + 누락 wav 경고

[열 구성]
  앞부분은 모든 클래스가 공유하는 공통 열, 뒤는 클래스별 부가 열이다.
  연장 데이터와 에너지 데이터의 열이 달라도 공통 열 순서가 같으므로 그대로
  합쳐진다. 부가 열이 없는 행은 공란이 된다.

[실행]
  python s4_build_master_metadata.py
=============================================================================
"""

import os
import csv
import glob
from collections import defaultdict

# 데이터 폴더 위치
#   코드 파일(src/)의 한 칸 위 = repo 폴더, 그 안의 data/ 를 쓴다.
#   C:\... 같은 절대경로를 쓰지 않는 이유: repo 를 다른 곳(예: OneDrive 밖)으로
#   옮겨도 코드를 고칠 필요가 없고, 어디서 실행하든 같은 폴더를 가리킨다.
#   data/ 는 .gitignore 에 들어 있어 GitHub 에 올라가지 않는다(음성 용량 문제).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data")

# 읽어올 metadata.csv 위치. data/ 아래를 깊이 상관없이 전부 찾는다.
#   data/prolong/metadata.csv, data/energy/v4/metadata.csv, data/normal/v4/metadata.csv ...
# 새 배치 폴더가 생겨도 여기를 고칠 필요가 없다. (밑줄 폴더는 아래에서 제외)
SOURCE_GLOBS = [
    os.path.join(DATA_ROOT, "**", "metadata.csv"),
]

MASTER_PATH = os.path.join(DATA_ROOT, "master_metadata.csv")
BY_CLASS_DIR = os.path.join(DATA_ROOT, "by_class")

# 클래스 표시 순서 (여기 없는 클래스는 뒤에 알파벳순으로 붙음)
CLASS_ORDER = ["normal", "prolong", "energy", "tremor"]

# 마스터 열 순서. 앞부분은 공통 열, 뒤는 클래스별 부가 열.
COMMON_FIELDS = [
    "row_no",            # 마스터에서 부여하는 통합 순번 (정렬 확인용)
    "new_filename", "old_filename", "class", "speaker_id", "gender", "speaker_f0_hz",
    "gender_flag", "source", "voice_name", "text_script", "qc_pass",
    "cluster_raw", "is_representative", "batch", "rel_path",
]
EXTRA_FIELDS = [
    "pattern",             # fade_out / fade_in / swell / dip / none
    "swing_db_target",     # 우리가 부여한 변동 폭
    "swing_db_measured",   # 실제 측정된 변동 폭 (라벨 검증·임계값 검토에 이 값을 쓴다)
    "trend_db",            # 문장 뒷부분 - 앞부분 (음수면 작아짐)
    "duration_sec", "sample_rate", "voice_id", "base_lufs", "seed",
    "use_cases", "lang_flag",        # lang_flag: ko / foreign (외국인 억양 화자 표시)
    "script_version", "generated_at",
]
MASTER_FIELDS = COMMON_FIELDS + EXTRA_FIELDS


def is_excluded_path(path: str) -> bool:
    """
    경로 어딘가에 밑줄로 시작하는 폴더가 있으면 통합에서 제외한다.
        _pilot_normal30   측정용 파일럿
        _test_batch1~3    과거 코드 테스트 결과물
        _archive_...      보관용 구 데이터
        _previews         미리듣기 음원 폴더
    폴더 이름만 보고 학습용/실험용을 구분할 수 있게 하는 규칙이다.
    """
    rel = os.path.relpath(path, DATA_ROOT)
    return any(part.startswith("_") for part in rel.split(os.sep)[:-1])


def find_sources():
    paths, seen, skipped = [], set(), []
    for pattern in SOURCE_GLOBS:
        for p in glob.glob(pattern, recursive=True):
            rp = os.path.normpath(p)
            if rp in seen or os.path.basename(rp) != "metadata.csv":
                continue
            if rp == os.path.normpath(MASTER_PATH):
                continue
            if is_excluded_path(rp):
                skipped.append(rp)
                continue
            seen.add(rp)
            paths.append(rp)

    if skipped:
        print("[제외] 밑줄로 시작하는 폴더는 통합하지 않습니다:")
        for p in sorted(skipped):
            print(f"   {os.path.relpath(p, DATA_ROOT)}")
    return sorted(paths)


def infer_batch(meta_path: str) -> str:
    """metadata.csv 가 들어있는 폴더 이름을 배치명으로 사용."""
    return os.path.basename(os.path.dirname(meta_path))


def parse_seq(filename: str) -> int:
    """energy_spk009_f_012.wav -> 12. 규칙에서 벗어나면 0."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def class_rank(cls: str) -> tuple:
    cls = (cls or "").strip().lower()
    if cls in CLASS_ORDER:
        return (0, CLASS_ORDER.index(cls), "")
    return (1, 0, cls)          # 미등록 클래스는 뒤로, 알파벳순


def load_all():
    rows, report = [], []
    for path in find_sources():
        batch = infer_batch(path)
        with open(path, newline="", encoding="utf-8-sig") as f:
            src_rows = list(csv.DictReader(f))

        audio_dir = os.path.join(os.path.dirname(path), "audio")
        for r in src_rows:
            out = {k: "" for k in MASTER_FIELDS}
            for k, v in r.items():
                if k in out:
                    out[k] = v
            # 파일명 열 이름이 다른 경우 보정
            if not out["new_filename"]:
                out["new_filename"] = r.get("filename", "") or r.get("file", "")
            if not out["class"]:
                out["class"] = r.get("label", "")
            if not out["batch"]:
                out["batch"] = batch
            # 실제 wav 를 찾아가기 위한 상대 경로 기록
            fn = out["new_filename"]
            cand = os.path.join(audio_dir, fn)
            base = audio_dir if os.path.exists(cand) else os.path.dirname(path)
            out["rel_path"] = os.path.relpath(os.path.join(base, fn), DATA_ROOT)
            rows.append(out)

        report.append((path, batch, len(src_rows)))

    print("[읽어온 소스]")
    if not report:
        print("  없음. DATA_ROOT 경로와 폴더 구조를 확인하세요.")
    for path, batch, n in report:
        print(f"  {n:>5}행  [{batch}]  {path}")
    return rows


def dedupe(rows):
    """new_filename 중복 제거 (나중에 읽은 쪽을 남김 = 최신 배치 우선)."""
    by_name = {}
    dup = 0
    for r in rows:
        key = r["new_filename"]
        if key in by_name:
            dup += 1
        by_name[key] = r
    if dup:
        print(f"[중복] new_filename 중복 {dup}건 -> 나중에 읽은 행만 남김")
    return list(by_name.values())


def sort_rows(rows):
    return sorted(rows, key=lambda r: (
        class_rank(r["class"]),
        r.get("speaker_id", ""),
        parse_seq(r["new_filename"]),
        r["new_filename"],
    ))


def write_master(rows):
    os.makedirs(os.path.dirname(MASTER_PATH), exist_ok=True)
    for i, r in enumerate(rows, start=1):
        r["row_no"] = i
    with open(MASTER_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[마스터] {len(rows)}행 -> {MASTER_PATH}")


def write_by_class(rows):
    os.makedirs(BY_CLASS_DIR, exist_ok=True)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["class"] or "unknown").strip().lower()].append(r)
    for cls, grp in groups.items():
        path = os.path.join(BY_CLASS_DIR, f"{cls}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=MASTER_FIELDS)
            w.writeheader()
            w.writerows(grp)
    print(f"[클래스별] {len(groups)}개 파일 -> {BY_CLASS_DIR}")


def print_summary(rows):
    """클래스 x 화자 교차표. 학습 균형/화자 분리 분할 가능성 점검용."""
    table = defaultdict(lambda: defaultdict(int))
    classes, speakers = set(), set()
    for r in rows:
        cls = (r["class"] or "unknown").strip().lower()
        spk = r.get("speaker_id") or "-"
        table[cls][spk] += 1
        classes.add(cls)
        speakers.add(spk)

    ordered_cls = sorted(classes, key=class_rank)
    ordered_spk = sorted(speakers)

    print("\n[클래스별 개수]")
    for cls in ordered_cls:
        print(f"  {cls:<10} {sum(table[cls].values()):>5}개")

    print("\n[클래스 x 화자 교차표]")
    header = "  " + "class".ljust(10) + "".join(s.rjust(9) for s in ordered_spk) + "    합계"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cls in ordered_cls:
        line = "  " + cls.ljust(10)
        line += "".join(str(table[cls].get(s, 0) or ".").rjust(9) for s in ordered_spk)
        line += str(sum(table[cls].values())).rjust(8)
        print(line)

    # 한 화자가 한 클래스에만 있으면 모델이 '화자'를 클래스로 학습할 위험
    risky = [s for s in ordered_spk
             if s != "-" and sum(1 for c in ordered_cls if table[c].get(s)) == 1]
    if risky:
        print(f"\n  [주의] 한 클래스에만 등장하는 화자: {risky}")
        print("        모델이 음향 패턴 대신 화자 특징을 외울 수 있습니다.")
        print("        가능하면 같은 화자로 여러 클래스를 생성하세요.")

    missing = [r["new_filename"] for r in rows
               if not os.path.exists(os.path.join(DATA_ROOT, r["rel_path"]))]
    if missing:
        print(f"\n  [주의] wav 파일을 찾지 못한 행 {len(missing)}건 (예: {missing[:3]})")


def main():
    rows = load_all()
    if not rows:
        return
    rows = sort_rows(dedupe(rows))
    write_master(rows)
    write_by_class(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
