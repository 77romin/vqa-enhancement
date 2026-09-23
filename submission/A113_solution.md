# A113_아잉~(AING) Solution

## 1. 솔루션 개요

본 솔루션은 4지선다형 Visual Question Answering(VQA) 문제를 위한 학습·검증·추론 파이프라인이다. 단순히 더 큰 모델로 교체하는 데 그치지 않고, 데이터 분할, 학습 목표, 추론 방식, 체크포인트, 제출 검증까지 평가 지표인 Accuracy에 맞게 재설계했다.

핵심 개선 사항은 다음과 같다.

- 전체 학습 데이터 사용
- SHA-256 및 perceptual hash(pHash) 기반 이미지 중복 감사
- 중복 이미지 그룹을 보존하는 leakage-safe train/validation split
- Qwen3.5-9B Vision-Language Model과 BF16 기반 LoRA 미세조정
- 질문·선택지 구간을 제외하고 assistant 정답 구간에만 loss를 적용하는 target masking
- 자유 생성 및 문자열 파싱 대신 `a/b/c/d` 후보의 조건부 log-probability 비교
- 선택지 순서 permutation ensemble을 통한 위치 편향 완화
- checkpoint 저장·재개, 원자적 추론 결과 저장, 제출 스키마 검증

발표 자료에 기록된 Kaggle Accuracy는 기존 `0.70837`에서 개선 후 `0.95829`이며, 절대 상승 폭은 `+0.24992`(24.992%p)이다. 다만 이 수치는 제공된 Kaggle 결과이며, 현재 노트북에는 해당 점수를 재현한 실행 로그와 ablation 결과가 저장되어 있지 않다. 따라서 개별 개선 요소의 기여도는 아직 확정된 인과 효과가 아니라 코드 차이와 학습 원리에 근거한 가설이다.

## 2. 문제 정의와 기존 방식의 한계

입력은 이미지, 질문, 네 개의 선택지 `a`~`d`이며 출력은 정답 선택지 한 글자이다. 평가지표는 Accuracy이다.

기존 파이프라인에서 확인한 주요 병목은 다음과 같다.

1. 전체 데이터가 아니라 약 200개 샘플만 사용해 학습 데이터가 지나치게 제한되었다.
2. `labels = input_ids.clone()` 방식으로 프롬프트 전체에 loss를 적용해 질문 복원과 정답 선택이 동시에 학습되었다.
3. 답을 자유롭게 생성한 뒤 문자열을 파싱해, 모델이 정답을 알아도 출력 형식 때문에 오답 처리될 수 있었다.
4. 동일하거나 유사한 이미지가 train과 validation에 동시에 들어가는지 충분히 검사하지 않았다.
5. 장시간 학습·추론 중 중단에 대비한 재개와 제출 전 자동 검증이 부족했다.

본 솔루션은 학습 신호와 의사결정 방식을 실제 평가 목표인 “네 후보 중 정답 선택”에 직접 맞추는 것을 중심 전략으로 삼았다.

## 3. 접근 전략

### 3.1 데이터 검증과 정답 구성

`precheck()`는 `train.csv`, `dev.csv`, `test.csv`, `sample_submission.csv`의 존재 여부와 필수 열을 검사한다. ID 결측·중복, 질문 및 선택지 결측, 이미지 파일 누락, 정답 범위, test와 sample submission의 행 수 및 ID 순서도 함께 검증한다.

dev 데이터의 `answer1`~`answer5`는 다수결로 단일 정답을 만든다. 동률이면 `a`, `b`, `c`, `d` 순으로 먼저 등장하는 답을 선택한다. 평가는 다수결 Accuracy를 주 지표로 사용하고, VQA soft score는 보조 진단 지표로 기록한다.

### 3.2 이미지 중복 감사와 leakage-safe split

이미지 중복은 두 단계로 탐지한다.

- SHA-256: 파일 바이트가 완전히 같은 이미지를 탐지한다.
- pHash: Hamming distance가 4 이하인 시각적으로 유사한 이미지를 탐지한다.

pHash 근접 검색에는 BK-tree를 사용하며, 탐지된 관계는 Union-Find로 전이적으로 합쳐 하나의 `image_group_id`를 만든다. 이후 `StratifiedGroupKFold`를 적용해 정답 분포를 최대한 유지하면서도 같은 이미지 그룹이 train과 validation에 동시에 포함되지 않게 한다. 기본 validation 비율은 약 15%이며 split seed는 `42`, `17`, `71`이다.

이 과정은 Kaggle 점수를 직접 높이는 장치라기보다, validation 누수로 인한 과대평가와 잘못된 모델 선택을 막는 장치다.

### 3.3 공통 프롬프트와 선택지 permutation

학습·검증·테스트에서 모두 `build_messages()`를 사용한다. 시스템 프롬프트는 작은 글자까지 주의 깊게 보도록 지시하고, 사용자 메시지는 이미지·질문·네 선택지를 같은 형식으로 구성한다.

학습 시 선택지 순서를 샘플별·epoch별로 결정적으로 섞고 정답 label을 새 위치에 맞게 변환한다. 이를 통해 “정답 내용”이 아니라 “항상 특정 위치나 특정 문자”를 고르는 shortcut을 줄인다.

### 3.4 모델과 LoRA 학습

기본 모델은 `Qwen/Qwen3.5-9B`이며 A100 80GB 환경을 기준으로 BF16을 사용한다. 이미지 입력은 최소 256, 최대 768 image token으로 제한하고, 최대 시퀀스 길이는 1,024 token으로 설정한다.

전체 모델을 미세조정하지 않고 language 영역의 attention projection 계층에만 LoRA를 적용한다. 기본 설정은 rank 16, alpha 32, dropout 0.05이다. `q_proj`, `k_proj`, `v_proj`, `o_proj` 등 실제로 존재하는 linear module만 동적으로 수집하며, 이름에 `visual` 또는 `vision`이 포함된 계층이 학습 가능 상태가 되면 즉시 오류를 발생시킨다. 즉 vision encoder는 동결하고 언어 추론부의 적응 파라미터만 학습한다.

### 3.5 Assistant target loss masking

`AnswerOnlyCollator`는 정답이 없는 prompt와 정답이 포함된 전체 대화를 각각 tokenization한다. 두 입력의 공통 prefix가 정확히 일치하는지 확인한 뒤 다음 구간을 `-100`으로 마스킹한다.

- system prompt
- 이미지 및 사용자 질문
- 네 개 선택지
- padding token

따라서 loss는 정답을 포함한 assistant target 구간에만 적용된다. 실제 active target이 존재하는지, truncation으로 정답이 사라지지 않았는지, decode한 target에 올바른 정답 문자가 포함되는지도 검사한다. 이 구현은 “정답 한 token만 학습”한다고 단정하기보다 “정답을 포함한 assistant target 구간에 loss를 집중한다”고 설명하는 것이 정확하다.

### 3.6 후보 log-probability 기반 추론

이 문제는 출력 후보가 `a`, `b`, `c`, `d`로 제한되어 있으므로 자유 생성이 필요하지 않다. `score_one_permutation()`은 동일한 prompt 뒤에 각 후보 문자를 붙인 네 입력을 만들고, 후보 token 구간의 평균 log-probability를 계산한다. 가장 높은 점수를 가진 원래 선택지를 최종 답으로 고른다.

추론 시에는 최대 네 개의 cyclic permutation을 사용할 수 있다. 각 permutation에서 얻은 점수를 원래 선택지 순서로 되돌린 뒤 평균해 위치 편향을 줄인다. 기본값은 두 번이며, `1`은 가장 빠르고 `4`는 위치 편향 완화가 가장 강하다.

이 방식의 장점은 다음과 같다.

- “정답은 b입니다” 같은 형식 차이로 인한 파싱 실패 제거
- 파서 fallback이 특정 label로 쏠리는 문제 제거
- 모든 후보의 상대 점수와 confidence 저장 가능
- 여러 모델의 후보 점수를 직접 평균하는 score ensemble 가능

### 3.7 학습과 모델 선택

기본 학습 설정은 다음과 같다.

| 항목 | 기본값 |
|---|---:|
| Epoch | 1 |
| Train batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Learning rate | `5e-5` |
| Weight decay | `0.01` |
| Warmup ratio | `0.05` |
| Max gradient norm | `1.0` |
| LoRA rank / alpha / dropout | `16 / 32 / 0.05` |
| Image token budget | 768 |
| Max sequence length | 1,024 |

Optimizer는 AdamW, scheduler는 cosine schedule을 사용한다. gradient accumulation의 마지막 묶음이 8개보다 작아도 실제 묶음 크기로 loss를 나누고 optimizer step을 수행한다. non-finite loss·gradient를 검사하고 gradient clipping을 적용한다.

각 epoch 뒤에는 validation loss가 아니라 후보 log-probability 추론의 Accuracy로 best adapter를 선택한다. 학습 중 빠른 비교는 최대 384개 균형 샘플로 수행하고, 최종 모델 평가는 별도의 `validate` 모드에서 전체 dev를 사용한다.

### 3.8 안정적인 추론과 제출 생성

test 추론 결과는 일정 간격마다 임시 파일에 먼저 쓴 뒤 `os.replace()`로 교체한다. 중단 후 기존 결과의 ID를 읽어 완료된 샘플을 건너뛸 수 있다. 개별 추론 오류는 별도 파일에 기록하고, 오류가 한 건이라도 있으면 제출 생성을 중단한다.

제출 직전에는 다음을 검사한다.

- prediction ID의 중복 및 결측 여부
- 답이 `a`, `b`, `c`, `d` 중 하나인지 여부
- test와 prediction의 ID 및 순서 일치 여부
- sample submission의 열이 정확히 `id`, `answer`인지 여부
- 최종 행 수와 답 결측 여부

검증을 통과한 제출 파일은 SHA-256 digest와 함께 저장한다.

## 4. 노트북 구성

| 노트북 구간 | 역할 |
|---|---|
| 1. Colab 환경 준비 | 의존성 설치 및 런타임 재시작 안내 |
| 1-1. Google Drive | Drive 마운트, 데이터 폴더 탐색, 암호화 ZIP 해제 |
| 2. 설정 | 실행 모드, 모델, 학습 및 추론 파라미터 지정 |
| 3. PRECHECK | CSV 스키마, ID, 이미지, 정답 및 제출 순서 검사 |
| 4. 데이터 감사 | 이미지 hash 감사와 group-safe split 생성 |
| 5. Prompt / Dataset | 공통 prompt, 선택지 permutation, loss collator |
| 6. Model / LoRA | model revision 확인, BF16 로드, LoRA 부착 |
| 7. 후보 점수 추론 | label별 조건부 log-probability 계산 |
| 8. 사전 검사 | zero-shot 평가와 loss-mask 검증 |
| 9. 학습 | 학습, checkpoint, 재개, best adapter 선택 |
| 10. 검증 / Final | 저장 adapter 평가 및 고정 step 전체 학습 |
| 11. Test 추론 | 중간 저장과 재개 가능한 test 추론 |
| 12. Ensemble / 제출 | score 평균과 최종 제출 스키마 검증 |
| 13. 실험 체크리스트 | 하이퍼파라미터 및 seed 비교 원칙 |

## 5. 실행 환경과 데이터 배치

### 5.1 권장 환경

- Google Colab
- NVIDIA A100 80GB
- Python 3
- BF16 지원 CUDA 환경
- Google Drive 또는 Colab 로컬 저장소

노트북은 Colab의 PyTorch/CUDA를 다시 설치하지 않는다. 첫 번째 설치 셀을 실행한 뒤에는 반드시 `런타임 > 세션 다시 시작`을 한 번 수행해야 한다.

### 5.2 필요한 데이터 구조

압축을 해제한 데이터 폴더에는 최소한 다음 파일과 각 CSV의 `path` 열이 가리키는 이미지가 있어야 한다.

```text
ssafy-16-2-ai/
├── train.csv
├── dev.csv
├── test.csv
├── sample_submission.csv
└── ... image files ...
```

기본 경로는 다음과 같다.

```python
DRIVE_DATA_DIR = Path("/content/drive/MyDrive/ssafy-16-2-ai")
DRIVE_ZIP_PATH = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
DATA_EXTRACT_ROOT = "/content"
```

Drive에 압축 해제된 폴더가 있으면 바로 사용한다. 폴더가 없고 ZIP만 있으면 비밀번호를 실행 시 입력받아 `/content`에 해제하며, 비밀번호는 노트북에 저장하지 않는다.

## 6. 실행 방법

### 6.1 기본 실행 절차

1. `A113_아잉~(AING).ipynb`를 Google Colab에서 연다.
2. 런타임 유형을 A100 GPU로 설정한다.
3. 패키지 설치 셀을 실행한다.
4. Colab 런타임을 한 번 재시작한다.
5. Google Drive에 데이터 폴더 또는 ZIP을 배치한다.
6. Drive 마운트 및 데이터 준비 셀을 실행한다.
7. 설정 셀의 `RUN_MODE`와 필요한 경로·파라미터를 지정한다.
8. 위에서 아래로 셀을 실행한다.

`RUN_MODE`만 바꿀 때에도 해당 설정 셀부터 아래 셀들을 다시 실행해야 한다. 서로 다른 실험은 가능하면 새 런타임에서 시작해 이전 모델 객체와 GPU 메모리 상태가 섞이지 않게 한다.

### 6.2 권장 실행 순서

#### 1) 데이터 및 규칙 검사

```python
RUN_MODE = "precheck"
```

필수 파일, 스키마, 이미지, ID와 대회 설정을 확인한다. 코드의 `RULES_CONFIRMED`, `EXTERNAL_PRETRAINED_ALLOWED`, `ENSEMBLE_ALLOWED`는 실제 대회 규정과 대조한 뒤 설정해야 한다.

#### 2) 이미지 감사와 split 생성

```python
RUN_MODE = "audit"
```

중복·유사 이미지 그룹과 seed별 split을 생성한다. `train` 모드에서도 split이 없으면 자동 실행되지만, 감사 보고서를 먼저 확인하기 위해 별도 실행을 권장한다.

#### 3) 학습 전 기준점 측정

```python
RUN_MODE = "zero_shot"
```

미세조정 전 전체 dev 성능을 저장한다. 이후 실험의 실제 개선 폭을 판단하는 기준점이다.

#### 4) Smoke test

```python
RUN_MODE = "smoke"
```

길이가 긴 샘플을 우선 사용해 20 optimizer step을 학습하고, checkpoint에서 재개해 25 step까지 진행한다. 이어서 valid 32개와 test 16개를 추론해 학습·저장·재개·추론 경로를 모두 점검한다.

#### 5) 본 학습

```python
RUN_MODE = "train"
```

seed 42 split으로 LoRA를 학습하고 validation Accuracy 기준 best adapter를 `/content/vqa_competition/runs/best/adapter`에 복사한다.

#### 6) 전체 dev 검증

```python
RUN_MODE = "validate"
INFER_ADAPTER_DIR = ADAPTER_DIR
```

저장한 adapter로 전체 dev를 평가한다. 이 단계의 결과를 기준으로 하이퍼파라미터와 최종 학습 step을 선택한다.

#### 7) 선택 사항: 전체 데이터 최종 학습

```python
RUN_MODE = "final"
FINAL_OPTIMIZER_STEPS = 검증에서_고정한_step_수
INCLUDE_DEV_IN_FINAL = False
```

검증이 끝난 뒤에만 사용한다. pretrained base에서 새로 시작하며, 검증된 optimizer step 수만큼 전체 train을 학습한다. dev를 최종 학습에 포함하는 것은 대회 규정상 허용되고 모든 모델 선택이 끝난 경우에만 고려한다.

#### 8) Test 추론

```python
RUN_MODE = "infer"
INFER_ADAPTER_DIR = ADAPTER_DIR  # 또는 final adapter 경로
```

후보 점수와 confidence를 포함한 test prediction 파일을 생성한다.

#### 9) 선택 사항: Ensemble

```python
RUN_MODE = "ensemble"
PREDICTION_FILES_FOR_ENSEMBLE = [
    "/path/to/model1_predictions.csv",
    "/path/to/model2_predictions.csv",
]
```

OOF 또는 dev에서 단일 모델보다 개선된 조합만 사용한다. 서로 같은 ID와 순서를 가진 prediction 파일의 `score_a`~`score_d`를 평균한다.

#### 10) 제출 파일 생성

```python
RUN_MODE = "submit_check"
PREDICTION_CSV = WORK_ROOT / "predictions" / "test_best.csv"
```

제출 스키마 검사를 통과하면 `/content/vqa_competition/submissions/submission_best.csv`를 생성한다.

### 6.3 주요 산출물

```text
/content/vqa_competition/
├── config/       # precheck, model revision 및 모델 설정
├── audit/        # 이미지 감사와 중복 그룹
├── splits/       # seed별 group-safe split
├── runs/         # config, checkpoint, adapter, metric, dev prediction
├── predictions/  # test 후보 점수와 ensemble 결과
└── submissions/  # 최종 제출 파일
```

## 7. 실험 설계

### 7.1 기본 원칙

- 한 번에 하나의 요인만 변경한다.
- 모든 비교에서 동일한 split, model revision, prompt, 평가 코드를 사용한다.
- zero-shot과 기존 baseline을 모두 기준점으로 남긴다.
- 단일 seed 탐색 후 최종 후보만 seed `17`, `42`, `71`로 반복한다.
- public leaderboard 변화만으로 모델, 하이퍼파라미터, ensemble weight를 선택하지 않는다.
- Accuracy뿐 아니라 학습 시간, 추론 시간, peak VRAM, 파싱 실패, confidence를 기록한다.

### 7.2 핵심 ablation

| 실험 | 고정 조건 | 변경 요소 | 확인 목적 |
|---|---|---|---|
| A | 3B / 기존 데이터 / 기존 추론 | assistant target masking 적용 | loss masking의 독립 효과 |
| B | 3B / masking 적용 / 기존 추론 | 약 200개에서 전체 train으로 확대 | 데이터 규모 증가 효과 |
| C | 전체 train / 동일 추론 | 3B 4-bit에서 9B BF16으로 변경 | 모델 규모·정밀도 효과 |
| D | 동일 모델과 동일 adapter | 자유 생성에서 후보 확률 비교로 변경 | 생성·파싱 오류 제거 효과 |

가능하면 각 단계는 바로 이전 단계와 비교해 누적 효과와 단독 효과를 구분한다. 여러 요소를 동시에 바꾼 결과만으로 특정 요소의 기여도를 단정하지 않는다.

### 7.3 하이퍼파라미터 탐색 순서

비용을 통제하기 위해 다음 순서로 한 축씩 탐색한다.

1. zero-shot 기준점
2. image token budget: `512 / 768 / 1024`
3. learning rate: `2e-5 / 5e-5 / 1e-4`
4. LoRA rank: `16 / 32`
5. permutation passes: `1 / 2 / 4`

앞 단계에서 선택한 값을 고정하고 다음 축을 비교한다. permutation passes는 Accuracy뿐 아니라 샘플당 추론 시간도 함께 측정한다.

### 7.4 반복 실험과 모델 선택

최종 후보는 세 seed에서 평가하고 다음 보수적 점수로 선택한다.

```text
selection_score = mean_accuracy - 0.5 × std_accuracy
```

평균 성능이 높더라도 seed에 따라 크게 흔들리는 모델은 불리하게 평가한다. 최종 보고에는 seed별 값, 평균, 표준편차, selection score를 모두 남긴다.

### 7.5 평가 항목

- 주 지표: dev majority Accuracy
- 보조 지표: VQA soft diagnostic
- 정답 class별 Accuracy
- OCR 포함 문제와 비-OCR 문제 성능
- 질문 또는 이미지 유형별 성능
- 후보 scoring confidence 분포
- 자유 생성 방식의 parsing failure 수
- 학습 시간, 추론 시간, peak reserved VRAM

### 7.6 Ensemble 검증

Ensemble은 test 점수를 보기 전에 OOF 또는 dev에서 weight를 고정한다. 먼저 동일 가중치 score 평균을 평가하고, 단일 최강 모델보다 개선되지 않으면 제출하지 않는다. 최종 제출은 검증 최강 단일 모델과 검증된 보수적 ensemble을 각각 보관하는 방식이 안전하다.

## 8. 재현성 확보 방법

각 실험마다 다음 항목을 함께 보존한다.

- model ID와 정확한 Hugging Face commit SHA
- 데이터 및 split 파일의 SHA-256
- 전체 config와 random seed
- 패키지 버전과 GPU 정보
- optimizer step, checkpoint, best adapter
- dev/test prediction과 후보별 score
- 평가 metric 및 peak VRAM
- 최종 submission의 SHA-256

현재 코드는 실행 시 `main`의 commit SHA를 조회하지만, 다른 날 완전히 같은 모델을 보장하려면 검증된 SHA를 `MODEL_REVISION`에 명시적으로 고정해야 한다.

## 9. 알려진 한계와 주의 사항

1. 현재 제공된 노트북은 실행 횟수와 출력이 저장되지 않은 상태다. `0.95829`의 재현 여부는 별도 실행으로 확인해야 한다.
2. 핵심 개선을 동시에 적용했기 때문에 각 요소의 독립 기여도는 ablation 전에는 확정할 수 없다.
3. `infer_resumable()`의 마지막 재정렬에서 문자열 ID로 `.loc`를 수행한다. 기존 prediction 또는 원본 ID가 숫자형이면 index dtype 불일치로 오류가 날 수 있으므로, 로드 직후 모든 데이터의 ID를 문자열로 통일하는 보완이 필요하다.
4. 학습 collator는 `MAX_LENGTH=1024`와 truncation을 사용하지만 후보 scoring 경로는 동일한 길이 제한을 명시하지 않는다. 매우 긴 샘플에서는 학습과 추론의 입력 조건이 달라질 수 있다.
5. checkpoint는 epoch validation 전에 저장되므로, 저장된 `best_metric`이 바로 뒤 validation 결과보다 한 단계 늦을 수 있다. 재개 시 best metric 보존을 엄밀히 하려면 validation 후 state를 다시 저장해야 한다.
6. 정규화된 질문 중복은 보고서에 기록하지만 현재 group split에는 반영하지 않는다. 동일 질문 누수가 중요한 데이터라면 이미지와 질문 그룹을 함께 고려해야 한다.
7. 후보 점수는 calibration되지 않은 상대 점수다. 저장되는 `confidence`를 실제 정답 확률로 해석하려면 별도의 calibration 실험이 필요하다.
8. 외부 pretrained 모델, dev의 최종 학습 사용, ensemble 허용 여부는 코드의 Boolean 값만 믿지 말고 실제 대회 규정과 다시 대조해야 한다.

## 10. 결론

본 솔루션의 핵심은 모델 크기 자체보다 데이터·학습 목표·추론 의사결정을 평가 방식에 일치시킨 것이다. 이미지 중복을 고려한 검증, assistant target 중심 loss, 후보 확률 기반 추론을 통해 성능과 신뢰성을 함께 개선하고, checkpoint·원자적 저장·제출 검증으로 장시간 실험의 실패 비용을 줄였다.

다음 목표는 단순히 높은 leaderboard 점수를 다시 얻는 것이 아니라, 고정된 model revision과 split, 실행 로그, ablation 결과를 함께 보존해 언제든 설명하고 재현할 수 있는 `0.95829`를 만드는 것이다.

## 11. 참고 파일

- `A113_아잉~(AING).ipynb`: 전체 학습·검증·추론 구현
- `A113_아잉~(AING).pdf`: 12페이지 발표 자료
- `presentation_script.md`: 발표 대본과 수치 해석상의 주의 사항
- `README.md`: 발표 자료 제출 형식 안내
