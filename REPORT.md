# FLOSS 논문 기반 Open-vocabulary Semantic Segmentation 스모크 테스트 재현 보고서

## 1. 실험 개요

선정 논문은 **"FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation"**이다. 본 과제에서는 논문의 전체 벤치마크를 재현하지 않고, 저장 공간과 CPU 실행 환경을 고려한 **스모크 테스트 재현**을 수행했다.

본 실험의 목적은 FLOSS의 핵심 아이디어인 클래스별 expert template 선택 및 fusion이 작은 샘플에서도 baseline averaged-template 방식과 비교 가능한 형태로 실행되는지 확인하는 것이다.

본 과제에서 AI 코딩 도구는 **Codex만 사용**했다.

## 2. 전체 재현이 아닌 스모크 테스트인 이유

원 저장소의 전체 평가 워크플로는 `tools/test.py`, `tools/eval_naclip.py`, 원본 dataset, checkpoint, 전체 validation set을 필요로 한다. 특히 전체 dataset 평가에는 Cityscapes 등 원본 데이터셋과 CLIP-DINOiser checkpoint가 필요하며, 실행 시간과 저장 공간 부담이 크다.

따라서 본 과제에서는 원본 notebook을 직접 실행하거나 수정하지 않고, precomputed feature를 사용하는 작은 실행 스크립트를 별도로 작성했다.

보존된 원본 notebook:

```text
notebooks/FLOSS_Reproducibility.ipynb
```

해당 파일은 변경하지 않았다.

## 3. 실행 방식 구분

### 3.1 Full dataset evaluation

Full dataset evaluation은 저장소의 원래 평가 방식이다. 예시는 다음과 같다.

```bash
python ./tools/test.py configs/clipdinoiser.py --dataset cityscapes
python ./tools/test.py configs/clipdinoiser.py --dataset cityscapes --mode fusion
```

이 방식은 전체 데이터셋, checkpoint, 긴 실행 시간, 더 큰 GPU/CPU 자원을 요구한다. 본 과제에서는 실행하지 않았다.

### 3.2 Dry-run

Dry-run 명령은 다음과 같다.

```bash
python scripts/run_floss_smoke_test.py --dry-run
```

Dry-run은 Hugging Face에서 어떤 파일을 선택할지 목록만 확인한다. feature 파일을 다운로드하지 않고, FLOSS 계산도 수행하지 않으며, metric 파일도 작성하지 않는다.

### 3.3 Smoke-test execution

실제 스모크 테스트 실행 명령은 다음과 같다.

```bash
python scripts/run_floss_smoke_test.py --device cpu
```

이 명령은 dry-run에서 확인된 16개 파일만 다운로드 또는 재사용하고, CPU에서 작은 FLOSS 비교 실험을 수행한다.

## 4. 입력 파일 및 다운로드 범위

Hugging Face repository:

```text
yasserben/floss-features
```

Dry-run 및 실제 실행에서 확인된 다운로드/재사용 파일 수는 총 16개이다.

```text
cityscapes_text_features.pt
vision_features_train/aachen_000000_000019_leftImg8bit.pt
vision_features_train/aachen_000001_000019_leftImg8bit.pt
vision_features_train/aachen_000002_000019_leftImg8bit.pt
vision_features_train/aachen_000003_000019_leftImg8bit.pt
vision_features_train/aachen_000004_000019_leftImg8bit.pt
vision_features_val/frankfurt_000000_000294_leftImg8bit.pt
vision_features_val/frankfurt_000000_000576_leftImg8bit.pt
vision_features_val/frankfurt_000000_001016_leftImg8bit.pt
vision_features_val/frankfurt_000000_001236_leftImg8bit.pt
vision_features_val/frankfurt_000000_001751_leftImg8bit.pt
gt_val/frankfurt_000000_000294_gtFine_labelTrainIds.png
gt_val/frankfurt_000000_000576_gtFine_labelTrainIds.png
gt_val/frankfurt_000000_001016_gtFine_labelTrainIds.png
gt_val/frankfurt_000000_001236_gtFine_labelTrainIds.png
gt_val/frankfurt_000000_001751_gtFine_labelTrainIds.png
```

원본 notebook의 broad `snapshot_download` 방식은 사용하지 않았다. 대신 `list_repo_files`와 `hf_hub_download`를 사용해 선택된 파일만 처리하도록 했다.

## 5. 생성된 결과 파일

실행 결과는 모두 다음 디렉터리에 저장했다.

```text
results/floss_smoke_test/
```

생성된 주요 파일은 다음과 같다.

```text
results/floss_smoke_test/download_manifest.json
results/floss_smoke_test/execution.log
results/floss_smoke_test/metrics.json
results/floss_smoke_test/per_class_iou.csv
results/floss_smoke_test/rankings_smoke.json
results/floss_smoke_test/scatter_mosaic.png
```

각 파일의 의미는 다음과 같다.

- `download_manifest.json`: 실제 선택된 Hugging Face 파일 목록
- `execution.log`: 실행 로그
- `metrics.json`: baseline 및 FLOSS mIoU와 per-class IoU
- `per_class_iou.csv`: class별 baseline/FLOSS IoU 비교표
- `rankings_smoke.json`: 5개 train feature로 계산한 entropy ranking
- `scatter_mosaic.png`: entropy ranking 시각화

## 6. 의존성 문제와 해결

초기 환경에는 실행에 필요한 일부 Python package가 없었다. 전체 `requirements.txt`는 설치하지 않았고, 명령 실행 중 실제로 누락된 것으로 확인된 package만 설치했다.

확인된 문제와 해결:

- `huggingface_hub` 누락: dry-run의 Hugging Face file listing을 위해 설치
- `numpy` 누락: runtime array 처리를 위해 설치
- `torch` 누락: CPU smoke-test 실행을 위해 CPU PyTorch 설치
- `Pillow` 누락: ground-truth PNG 로딩을 위해 설치
- `matplotlib` 누락: `scatter_mosaic.png` 생성을 위해 설치

설치 과정에서 sandbox DNS 제한으로 인해 일부 pip 명령은 network escalation이 필요했다. 추가적인 전체 dependency 설치는 하지 않았다.

## 7. 실험 설정

스모크 테스트 설정:

```text
DATASET = cityscapes
NUM_IMAGES = 5
NUM_VAL_IMAGES = 5
device = cpu
ranking_resolution = (512, 256)
```

이 설정은 논문 수치 재현을 목표로 한 것이 아니라, 제한된 compute/storage 환경에서 baseline과 FLOSS fusion의 실행 가능성과 결과 생성 과정을 검증하기 위한 것이다.

## 8. 결과

`results/floss_smoke_test/metrics.json`에서 확인된 결과는 다음과 같다.

| Method | mIoU |
|---|---:|
| Baseline averaged-template | 20.3923 |
| FLOSS expert-fusion | 23.3612 |
| Delta | +2.9689 |

작은 5-image validation subset에서 FLOSS expert-fusion 결과가 averaged-template baseline보다 `+2.9689` mIoU 높게 나왔다.

주의할 점은 일부 class에 대해 validation subset 내 ground truth 또는 prediction 분포가 충분하지 않아 `NaN` IoU가 포함되어 있다는 것이다. 이 결과는 전체 Cityscapes benchmark 결과로 해석하면 안 된다.

## 9. 한계

본 실험의 한계는 다음과 같다.

- train feature 5개와 validation feature 5개만 사용했다.
- ranking resolution을 CPU 실행을 위해 낮췄다.
- 원본 논문의 전체 dataset benchmark와 동일한 조건이 아니다.
- raw Cityscapes dataset이나 CLIP-DINOiser checkpoint를 사용한 full evaluation을 수행하지 않았다.
- 결과는 smoke-test sanity check로만 해석해야 한다.

## 10. 결론

본 과제에서는 FLOSS 저장소를 inspection한 뒤, 원본 notebook을 수정하지 않고 CPU 및 저장 공간 제약에 맞춘 작은 실행 워크플로를 구성했다. Dry-run으로 다운로드 대상을 먼저 검증했고, 실제 실행에서는 확인된 16개 파일만 다운로드/재사용했다.

스모크 테스트 결과, 5개 validation feature 기준으로 baseline averaged-template mIoU는 `20.3923`, FLOSS expert-fusion mIoU는 `23.3612`, 차이는 `+2.9689`였다.

이 결과는 논문 전체 재현이 아니라, FLOSS 방식의 최소 실행 가능성과 artifact 생성 과정을 검증한 과제용 smoke-test reproduction이다.

