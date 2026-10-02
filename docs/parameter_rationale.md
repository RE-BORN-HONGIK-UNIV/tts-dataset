# synthetic_v1 파라미터 및 판단 근거

## 1. 문서 목적
본 문서는 synthetic_v1 데이터 생성과 품질관리에서 사용한 파라미터,
연장 표기 규칙, 판단 기준의 근거와 한계를 기록한다.

외부 연구에서 직접 검증된 값, 프로젝트 내부 pilot/QC에서 정한 잠정값,
아직 확정하지 않은 운영값을 명확히 구분한다.

## 2. 연장(prolongation)의 해석

### 학술적 배경
연장은 일반적으로 발화 단위의 지속시간 증가로 관찰될 수 있으나,
절대 지속시간 하나만으로 판단하기보다 주변 발화와의 상대적 길이,
연속성, 발화 흐름을 함께 고려할 필요가 있다.

- Gallo, E., Schettino, L., & Cutugno, F. (2024).
  “About theee hesitant voice: Acoustic and functional analysis of
  prolongations in Italian spontaneous speech.”
  ISCA Archive.
  https://www.isca-archive.org/lw_2024/gallo24_lw.html

반복, 연장, 막힘 등은 서로 구별되는 비유창성 양상으로 기술된다.

- American Speech-Language-Hearing Association (ASHA).
  “Stuttering, Cluttering, and Fluency.”
  https://www.asha.org/practice-portal/clinical-topics/fluency-disorders/

### 프로젝트 적용 한계
- 위 문헌은 본 프로젝트의 한국어 TTS 입력 표기, B/C 단계, 71명 화자,
  특정 문장 위치를 직접 검증한 연구가 아니다.
- 따라서 B/C 표기 및 생성 위치는 임상 임계값이나 보편적 음성 기준이 아니라,
  다화자 파일럿 청취 QC에 기반한 프로젝트 내부 잠정 생성 규칙이다.

## 3. B/C 연장 표기

| 항목 | 현재 정의 | 근거 상태 |
|---|---|---|
| B | 프로젝트 내부의 상대적으로 약한 연장 표기 단계 | 파일럿·청취 QC 기반 잠정값 |
| C | 프로젝트 내부의 상대적으로 강한 연장 표기 단계 | 파일럿·청취 QC 기반 잠정값 |
| 실제 지속시간 | TTS 화자·문장·엔진 출력에 따라 달라짐 | 고정값 미설정 |
| 합격 기준 | 청취상 목표 위치의 상대적 길이 증가와 연속성 확인 | 운영적 QC 기준 |

> B/C는 실제 사람 발화의 지속시간이나 불안도 수준을 수치화한 등급이 아니다.

## 4. 파일럿 기반 규칙 선정

### 파일럿 개요
- 파일럿 화자: `spkS006`, `spkS009`, `spkS010`
- 평가 파일 수: 33개
- 최종 결과: pass 25개, exclude 8개, review 0개

### 1차 대량 생성 포함 규칙
- 세 화자에서 청취상 통과한 7개 규칙을 사용한다.
- 규칙 목록은 `docs/data_generation.md`의 “확정 연장 규칙”과 동일해야 한다.

### 제외 규칙
- `sent_03 target_02`, `sent_04 target_01`, `sent_04 target_02`:
  일부 화자에서 반복·분절 또는 합성 오류가 확인됨
- `sent_06 target_02`:
  연장 인지가 약하거나 분절 가능성이 확인됨

## 5. 후속 검증 원칙
- 대량 생성 후 기술 QC와 청취 QC를 분리 수행한다.
- TTS 엔진, 목소리 모델, 문장 또는 표기 규칙이 바뀌면 이전 pilot 결과를
  그대로 일반화하지 않고 재검증한다.
- 향후 실제 참여자 음성을 수집·분석한다면, 연구 윤리, 동의 절차,
  개인정보·민감정보 보호, 임상적 해석의 제한을 별도로 설계한다.