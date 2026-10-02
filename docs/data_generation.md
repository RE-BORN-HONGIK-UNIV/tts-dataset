# synthetic_v1 데이터 생성 설계

## 1. 목적
본 데이터셋은 고립청년 대상 대면 면접 훈련 서비스에서 사용할
음성 특성 분석 기능의 개발·검증을 위한 합성 음성 데이터다.

본 문서의 prolongation 라벨은 TTS 입력 텍스트에 특정 연장 표기를 적용해
생성한 합성 라벨이며, 실제 개인의 불안도·정신건강 상태·면접 역량 또는
임상적 유창성 장애를 의미하거나 진단하지 않는다.

## 2. 버전과 범위
- 데이터 버전: synthetic_v1
- 생성 대상: 승인 화자 71명
- 화자당 연장 파일 수: 7개
- 계획 생성 수: 497개
- 음성 생성 방식: Typecast TTS
- 정상 원본 수정 여부: 수정하지 않음

## 3. 입력과 출력
### 입력
- `metadata/speakers_approved_71.csv`
- `.env`의 Typecast API 인증 정보
- `src/v1/generate_prolongation_v1.py`의 `PROLONGATION_RULES`

### 출력
- `audio/prolongation/`
- `metadata/prolongation_v1_manifest.csv`
- `metadata/prolongation_v1_generation_log.csv`

## 4. 파일명 규칙
```text
spkS###__sent_##__target_##__B_or_C.wav
```

예시:
```text
spkS006__sent_01__target_01__C.wav
```

| 항목 | 의미 |
|---|---|
| `spkS###` | 합성 화자 식별자 |
| `sent_##` | 문장 식별자 |
| `target_##` | 문장 내 연장 적용 목표 위치 |
| `B` 또는 `C` | 프로젝트 내부 연장 표기 단계 |

## 5. 확정 연장 규칙
| sentence_id | target_id | 목표 단어 또는 구 | 표기 단계 | 선정 상태 |
|---|---|---|---|---|
| sent_01 | target_01 | 배우고 | C | 생성 |
| sent_02 | target_01 | 살펴보겠습니다 | B | 생성 |
| sent_02 | target_02 | 우선 | B | 생성 |
| sent_03 | target_01 | 마무리하겠습니다 | C | 생성 |
| sent_05 | target_01 | 차분한 | C | 생성 |
| sent_05 | target_02 | 마음으로 | C | 생성 |
| sent_06 | target_01 | 이번 | C | 생성 |

> 실제 TTS 입력 문자열은 코드의 `PROLONGATION_RULES`를 기준으로 관리한다.
> 표와 코드가 다를 경우 코드 실행을 중단하고 불일치 원인을 먼저 수정한다.

## 6. 제외·보류 규칙
- `sent_03 target_02`, `sent_04 target_01`, `sent_04 target_02`:
  파일럿 청취에서 반복·분절 또는 합성 오류가 확인되어 1차 대량 생성에서 제외했다.
- `sent_06 target_02`:
  연장 인지가 약하거나 분절 가능성이 확인되어 제외했다.
- 제외 규칙은 폐기하지 않으며, TTS 표기 변경 또는 음성 엔진 변경 시 재시험 후보로 관리한다.

## 7. 재현 방법
```bash
python src/v1/generate_prolongation_v1.py
python src/v1/generate_prolongation_v1.py --run
```

- 첫 번째 명령은 생성 계획 및 manifest를 점검한다.
- 두 번째 명령은 실제 API 호출 및 WAV 저장을 수행한다.
- 이미 정상 생성된 파일은 skip한다.
- 생성 이후에는 `docs/qc_protocol.md`에 따라 QC를 수행한다.