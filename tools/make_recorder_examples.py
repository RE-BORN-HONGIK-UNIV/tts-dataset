"""
지인 녹음 웹(reborn-voice-collect)의 "예시 듣기" 음성 4개를 Typecast Siyeon 목소리로 만든다.
연장은 늘여 쓴 글자("저어어엉말")를 TTS가 그대로 읽게 한다(WORLD 가공은 끊김·잡음이 나서 쓰지 않음).
작아짐은 슬픔 감정(0.8)으로 합성한 뒤 기존 fade_out(cosine dB 감쇠)을 적용한다.
실행: .venv\\Scripts\\python tools\\make_recorder_examples.py [이름...]
"""
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
OUT = Path(r"C:\Users\seoyn\RebornProject\reborn-voice-collect\examples")
VOICE_ID = "tc_6568164fe05ddffee8b0e271"  # spkS040 Siyeon
FADE_DB = 24.0  # 예시 전용: 폰 스피커로도 작아지는 게 들리게(학습 데이터 fade는 12~18dB)
SAD = {"emotion_type": "preset", "emotion_preset": "sad", "emotion_intensity": 0.8}  # 1.0은 말끝이 늘어져 연장처럼 들림

# 이름 → (문장, 감정 옵션, 작아짐 적용 여부). 짧은 면접 문장으로, 대본 8세트와 겹치지 않게.
# 연장은 자신 없는 머뭇거림(sad 0.5). smart 프롬프트(당황 상황)보다 자연스러웠다.
UNSURE = {"emotion_type": "preset", "emotion_preset": "sad", "emotion_intensity": 0.5}
LINES = {
    "normal": ("안녕하세요, 잘 부탁드립니다.", None, False),
    "fade": ("그 부분은 아직 부족한 것 같습니다.", SAD, True),
    "prolong_front": ("저어어어는 협업을 잘합니다.", UNSURE, False),  # 후보 중 UNSURE 채택
    # 단어 가운데 글자를 끈다(꾸[준]히). 열심히·책임감 후보보다 자연스러웠다.
    "prolong_mid": ("저는 꾸주우우운히 노력합니다.", UNSURE, False),
}


# 한 문장 합성(기존 normal 생성과 같은 조건, 감정 옵션 선택).
def synth(client, text, prompt=None):
    from typecast.models import LanguageCode, Output, TTSRequest
    req = TTSRequest(text=text, model="ssfm-v30", voice_id=VOICE_ID, language=LanguageCode.KOR,
                     output=Output(target_lufs=-18, audio_pitch=0, audio_tempo=1.0, audio_format="wav"),
                     **({"prompt": prompt} if prompt else {}))
    audio, sr = sf.read(io.BytesIO(client.text_to_speech(req).audio_data))
    return (audio.mean(axis=1) if audio.ndim > 1 else audio).astype(np.float64), sr


# 기존 fade_out(cosine dB 감쇠)을 제곱해 앞은 거의 그대로, 뒤에서 확 작아지게 한다(예시 전용).
def fade_out(audio, strength_db):
    ease = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, len(audio)))) ** 2
    return audio * np.power(10.0, -(strength_db * ease) / 20.0)


# wav로 쓰고 mp3(모노 64kbps)로 바꾼다.
def save(name, audio, sr):
    wav = OUT / f"{name}.wav"
    sf.write(wav, audio.astype(np.float32), sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-ac", "1", "-b:a", "64k",
                    str(OUT / f"{name}.mp3")], check=True)
    wav.unlink()
    print(f"{name}: {len(audio) / sr:.1f}s")


def main():
    load_dotenv(REPO / ".env")
    from typecast import Typecast
    client = Typecast()  # .env의 TYPECAST_API_KEY를 읽는다
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])  # 이름을 주면 그것만 다시 만든다
    for name, (text, prompt, fade) in LINES.items():
        if only and name not in only:
            continue
        audio, sr = synth(client, text, prompt)
        save(name, fade_out(audio, FADE_DB) if fade else audio, sr)


if __name__ == "__main__":
    main()
