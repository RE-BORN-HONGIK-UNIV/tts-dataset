"""synthetic_v2: v1에 없던 새 목소리로 같은 문장을 연장·보통·천천히 세 가지로 읽혀 짝을 만든다.
연장은 늘인 글자("주우우우운")를 Typecast가 읽게 하고, 글자별 시간 정보로 연장 구간을 기록한다(손으로 표시하지 않음).
사용: .venv\\Scripts\\python src\\v2\\synth_v2.py screen            (목소리 선별용 한 문장씩)
      .venv\\Scripts\\python src\\v2\\synth_v2.py make <태그> <목소리 수> <목소리당 문장 수>
"""
import csv
import io
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
OUT = Path(r"G:\내 드라이브\tts_dataset\synthetic_v2")
SCREEN_TEXT = "안녕하세요. 저는 오늘 면접을 보러 왔습니다."
EMOTIONS = [("normal", 1.0), ("sad", 0.5), ("tonedown", 0.5)]  # sad 1.0은 말끝이 늘어져 연장처럼 들려서 0.5까지만
SLOW = 0.85
EXTRA_MIN, EXTRA_MAX = 0.3, 2.5  # 보통판보다 늘어난 초. 이 밖이면 연장 실패(안 늘어남·폭주)로 보고 다시 만든다
SIMPLE = {2: 0, 3: 1, 6: 4, 7: 5, 9: 0, 10: 1, 11: 5, 12: 8, 14: 4, 15: 5, 16: 20, 17: 13, 19: 20}  # 겹모음 → 끌 때 남는 홑모음
GAP_DB, GAP_MS = 20, 40  # 시험 생성 20개로 맞춤: 귀로 '끊김' X 5개 중 4개를 잡고 O 15개는 하나도 안 걸림
LOCK = threading.Lock()


def stretch(syl, k):
    """한 글자를 k글자만큼 늘여 쓴다: 준 → 주우우우운(k=4), 저 → 저어어어어."""
    c = ord(syl) - 0xAC00
    cho, jung, jong = c // 588, c % 588 // 28, c % 28
    mk = lambda i, j, t=0: chr(0xAC00 + i * 588 + j * 28 + t)
    v = SIMPLE.get(jung, jung)
    return mk(cho, jung) + mk(11, v) * (k - 1) + mk(11, v, jong)


def texts(line, k):
    """'[저]는 ...' → (연장 문장, 보통 문장, 대상 글자 위치)."""
    pre, rest = line.split("[")
    syl, post = rest.split("]")
    end = "?" if post.endswith("까요") else "."
    return pre + stretch(syl, k) + post + end, pre + syl + post + end, len(pre)


def voices_unused():
    """v1 선별·시험에 쓴 목소리와 제외 표시가 있는 목소리를 뺀 후보 목록."""
    rows = lambda p: list(csv.DictReader(open(REPO / p, encoding="utf-8-sig")))
    used = {r["voice_id"] for r in rows("data/_speaker_screening/selected_speakers.csv")}
    used |= {r["voice_id"] for r in rows("data/_pilot_normal30/metadata.csv")}
    used.add("tc_6568164fe05ddffee8b0e271")  # 녹음 웹 예시에 쓴 Siyeon
    return [r for r in rows("data/voice_candidates.csv") if r["voice_id"] not in used and not r["excluded_by"]]


def synth(client, text, voice_id, emotion, tempo):
    """한 문장을 합성해 (wav 바이트, 글자별 시간 목록)을 돌려준다. 실패하면 두 번 더 시도한다."""
    from typecast.models import LanguageCode
    from typecast.models.tts import Output, PresetPrompt, TTSRequestWithTimestamps
    req = TTSRequestWithTimestamps(text=text, model="ssfm-v30", voice_id=voice_id, language=LanguageCode.KOR,
                                   prompt=PresetPrompt(emotion_preset=emotion[0], emotion_intensity=emotion[1]),
                                   output=Output(target_lufs=-18, audio_tempo=round(tempo, 3), audio_format="wav"))
    for attempt in range(3):
        try:
            r = client.text_to_speech_with_timestamps(req, granularity="char")
            return r.audio_bytes, [(ch.text, ch.start, ch.end) for ch in r.characters]
        except Exception:
            if attempt == 2:
                raise


def gap_ms(wav, s, e):
    """연장 구간 안에서 소리가 끊긴(가장 큰 소리보다 GAP_DB 이상 작은) 가장 긴 시간(ms). 늘인 모음이 중간에 끊기는 목소리를 거른다."""
    y, sr = sf.read(io.BytesIO(wav))
    y = y.mean(1) if y.ndim > 1 else y
    h = sr // 100
    db = 20 * np.log10(np.sqrt((y[:len(y) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9)
    run = best = 0
    for low in db[int(s * 100) + 2:int(e * 100) - 2] < np.percentile(db, 90) - GAP_DB:
        run = run + 1 if low else 0
        best = max(best, run)
    return best * 10


def span(chars, text, i, n):
    """text[i:i+n] 글자들의 시작·끝 초. 시간 목록이 글자와 하나씩 맞지 않으면 None."""
    if "".join(c[0] for c in chars) != text:
        return None
    return chars[i][1], chars[i + n - 1][2]


def append(path, row):
    """목록 표에 한 줄을 더한다(처음이면 머리줄도)."""
    with LOCK:
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


def screen(client):
    """후보 목소리마다 같은 한 문장을 보통으로 읽혀 저장한다(Whisper 선별용)."""
    d = OUT / "screen"
    d.mkdir(parents=True, exist_ok=True)

    def one(v):
        wav = d / f"{v['voice_name']}.wav"
        if not wav.exists():
            wav.write_bytes(synth(client, SCREEN_TEXT, v["voice_id"], ("normal", 1.0), 1.0)[0])
            append(d / "screen_list.csv", {k: v[k] for k in ["voice_id", "voice_name", "gender", "use_cases"]} | {"text": SCREEN_TEXT})
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, voices_unused()))


def make_one(client, d, v, s, rng):
    """한 목소리·한 문장으로 연장·보통·천천히 세 파일을 만들고 목록 표에 적는다."""
    emo, tempo = rng.choice(EMOTIONS), rng.uniform(0.92, 1.08)
    stem = f"{v['voice_name']}__{s['sentence_id']}"
    if (d / "audio" / f"{stem}__slow.wav").exists():
        return
    nor_text = texts(s["text"], 1)[1]
    nor_wav, nor_chars = synth(client, nor_text, v["voice_id"], emo, tempo)
    base = {"voice_id": v["voice_id"], "voice_name": v["voice_name"], "gender": v["gender"], "sentence_id": s["sentence_id"],
            "target_type": s["target_type"], "emotion": emo[0], "intensity": emo[1], "tempo": round(tempo, 3)}
    i = texts(s["text"], 1)[2]
    ns = span(nor_chars, nor_text, i, 1)
    k, status = rng.randint(3, 5), "fail"
    for _ in range(3):  # 안 늘어나면 더 길게, 폭주하거나 끊기면 더 짧게 다시
        pro_text = texts(s["text"], k)[0]
        pro_wav, pro_chars = synth(client, pro_text, v["voice_id"], emo, tempo)
        ps = span(pro_chars, pro_text, i, k + 1)
        extra = (ps[1] - ps[0]) - (ns[1] - ns[0]) if ps and ns else None
        gap = gap_ms(pro_wav, *ps) if ps else None
        if extra is not None and EXTRA_MIN <= extra <= EXTRA_MAX and gap < GAP_MS:
            status = "ok"
            break
        k = k + 1 if extra is not None and extra < EXTRA_MIN else max(2, k - 1)
    slow_wav, _ = synth(client, nor_text, v["voice_id"], emo, tempo * SLOW)
    now = datetime.now().isoformat(timespec="seconds")
    for kind, wav, text, sp, extra_ in [("prolong", pro_wav, pro_text, ps, extra), ("normal", nor_wav, nor_text, ns, None),
                                        ("slow", slow_wav, nor_text, None, None)]:
        f = d / "audio" / f"{stem}__{kind}.wav"
        f.write_bytes(wav)
        append(d / "manifest.csv", base | {"kind": kind, "file": f.name, "text": text, "k": k if kind == "prolong" else "",
                                            "span_s": round(sp[0], 3) if sp else "", "span_e": round(sp[1], 3) if sp else "",
                                            "extra_sec": round(extra_, 3) if extra_ is not None else "",
                                            "gap_ms": gap if kind == "prolong" else "",
                                            "status": status if kind == "prolong" else "ok", "created": now})


def make(client, tag, n_voices, per_voice):
    """선별된 목소리 중 귀 확인에서 뺀 목소리를 제외하고 앞에서 n명을 골라, 목소리마다 front·mid 문장을 반씩 돌아가며 배정해 만든다."""
    d = OUT / tag
    (d / "audio").mkdir(parents=True, exist_ok=True)
    rows = lambda name: list(csv.DictReader(open(OUT / "screen" / name, encoding="utf-8-sig")))
    out = {r["voice_id"] for r in rows("voice_review.csv") if r["verdict"] == "빼기"}
    voices = [v for v in rows("selected_voices.csv") if v["voice_id"] not in out][:n_voices]
    sents = list(csv.DictReader(open(REPO / "src" / "v2" / "sentences_v2.csv", encoding="utf-8-sig")))
    front = [s for s in sents if s["target_type"] == "front"]
    mid = [s for s in sents if s["target_type"] == "mid"]
    jobs = []
    for vi, v in enumerate(voices):  # 목소리마다 시작 위치를 밀어 문장마다 읽는 목소리 수를 고르게 한다
        h = per_voice // 2
        picks = [front[(vi * h + j) % len(front)] for j in range(h)] + [mid[(vi * h + j) % len(mid)] for j in range(per_voice - h)]
        jobs += [(v, s, random.Random(f"{v['voice_id']}|{s['sentence_id']}")) for s in picks]
    with ThreadPoolExecutor(4) as ex:
        for fut in [ex.submit(make_one, client, d, *j) for j in jobs]:
            fut.result()
    print(f"{tag}: {len(jobs)}개 문장 × 3")


if __name__ == "__main__":
    assert stretch("준", 4) == "주우우우운" and stretch("저", 4) == "저어어어어" and stretch("협", 3) == "혀어어업"
    load_dotenv(REPO / ".env")
    from typecast import Typecast
    client = Typecast()  # .env의 TYPECAST_API_KEY
    if sys.argv[1] == "screen":
        screen(client)
    else:
        make(client, sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
