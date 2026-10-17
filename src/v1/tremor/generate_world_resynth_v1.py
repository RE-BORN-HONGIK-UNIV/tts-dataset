"""tremor v1과 같은 원본 420개를 흔들림 크기 0으로 WORLD 재합성한다.
떨림 모델의 '아님' 쪽에도 WORLD 흔적을 넣어 "WORLD 소리 = 떨림" 지름길을 끊기 위한 데이터다.
합성 과정은 generate_tremor_v1.py를 그대로 쓰고 변조 깊이만 0으로 둔다.
"""

import sys

import numpy as np
import soundfile as sf

from generate_tremor_v1 import (
    EXPECTED_OUTPUT_COUNT, OUTPUT_SUBTYPE, TremorCondition, discover_sources,
    get_paths, is_valid_wav, read_mono_float64, synthesize_tremor, write_csv,
)

NO_TREMOR = TremorCondition("world_resynth", rate_hz=5.0, f0_depth_semitones=0.0,
                            amplitude_depth=0.0, amplitude_phase_rad=0.0)


def main() -> None:
    """원본 420개를 재합성해 저장하고 매니페스트를 남긴다."""
    paths = get_paths()
    root = paths["output_dir"].parent.parent
    out_dir = root / "audio" / "world_resynth_normal"
    manifest = root / "metadata" / "world_resynth_normal_manifest.csv"
    if out_dir.exists() or manifest.exists():
        sys.exit(f"[오류] 이미 결과가 있어 덮어쓰지 않습니다: {out_dir}")

    sources, _ = discover_sources(paths["normal_dir"])
    out_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)  # 깊이 0이라 난수는 결과에 영향 없음
    rows = []
    for i, (src, spk, sent) in enumerate(sources, 1):
        audio, sr = read_mono_float64(src)
        output, _ = synthesize_tremor(audio, sr, NO_TREMOR, rng)
        out = out_dir / f"{src.stem}__world_resynth.wav"
        sf.write(out, output, sr, subtype=OUTPUT_SUBTYPE)
        assert is_valid_wav(out), out
        rows.append({"input_filename": src.name, "speaker_id": spk, "sentence_id": sent,
                     "output_filename": out.name, "sample_rate_hz": sr})
        print(f"[완료] {i}/{len(sources)} {out.name}")

    assert len(rows) == EXPECTED_OUTPUT_COUNT
    write_csv(manifest, rows)
    print(f"저장: {out_dir} ({len(rows)}개), {manifest}")


if __name__ == "__main__":
    main()
