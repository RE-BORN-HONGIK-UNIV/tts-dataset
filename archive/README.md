# archive/

`synthetic_v1` 최종 파이프라인을 정하기까지 사용한 **파일럿·탐색 스크립트 보관소**입니다.
당시 실행한 코드를 그대로 남겨 `docs/experiment_log.md`의 기록과 대조할 수 있게 했습니다.

> 실행용이 아닙니다. 대부분 당시 작업 PC의 절대경로(`C:\Users\...`, `G:\내 드라이브\...`)를
> 하드코딩하고 있고, 결과물은 이미 생성되어 기록에 반영되었습니다.
> 최종 데이터셋을 다시 만들 때는 `src/v1/` 아래 스크립트를 사용합니다.

| 폴더 | 내용 | 대체한 최종 스크립트 |
|---|---|---|
| `pilots_pre_v1/` | v1 이전 단계(s1~s4). Typecast 화자 목록 조회, normal 30개 파일럿, 화자 청취 스크리닝, 에너지 파일럿 20쌍과 QC, 스크리닝 판정 반영(`apply_screening_decisions.py`) | `src/v1/normal/`, `src/v1/energy/` |
| `prolongation_exploration/` | 연장(prolongation) 생성 방식 탐색. 실제·합성 음성 유성 구간 비교, 텍스트 기반 연장 파일럿, 다화자 파일럿과 청취 QC 기록 저장 | `src/v1/prolongation/` |
| `v1_pilots/` | tremor 조건 탐색 파일럿 round1~4(make/qc), 1차 tremor QC(`qc_tremor_v1.py`, v2로 대체), slow_normal 속도 파일럿 2개 | `src/v1/tremor/`, `src/v1/slow_normal/` |

## 참고

- `pilots_pre_v1/s3_generate_energy.py`는 `s4_generate_energy_pilot.py`로 대체되어 결과물을 남기지 않았습니다.
- `pilots_pre_v1/s4_build_master_metadata.py`는 v1 이전 로컬 `data/` 구조용이며 v1에서는 사용하지 않았습니다.
- `pilots_pre_v1/` 스크립트는 저장소 루트의 `data/`를 기준으로 경로를 계산합니다(`archive/pilots_pre_v1/`의 두 칸 위).
- `prolongation_exploration/save_multispeaker_pilot_qc.py`가 만든 청취 QC 기록은 `metadata/prolong_multispeaker_pilot_qc.csv`에 있고,
  이 기록을 바탕으로 `src/v1/prolongation/generate_prolongation_v1.py`의 생성 규칙을 정했습니다.
