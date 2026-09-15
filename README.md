# main : 사람 쓰러짐 감지 팀 프로젝트
# solo : 실종자 탐지 서비스 개인 프로젝트

## 보안 업데이트 후 설치

Python 3.12의 새 가상환경에서 저장소 루트 기준으로 설치합니다.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade "pip>=26.2.1"
.venv\Scripts\python -m pip install -r main/Code/requirements/requirements.txt
# YOLOv7 학습 및 TensorBoard를 사용하는 경우
.venv\Scripts\python -m pip install -r main/Code/yolov7/requirements.txt
.venv\Scripts\python tests/security_smoke.py
```

Linux/macOS에서는 `.venv/bin/python`을 사용합니다. Conda 사용 시
`main/Code/requirements` 폴더에서 `conda env create -f environment.yml`을 실행합니다.
GPU 설치는 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서
드라이버에 맞는 빌드를 선택하되, `torch==2.13.0`과 `torchvision==0.28.0` 조합을 유지합니다.

### 기존 모델 파일 주의사항

PyTorch 2.6부터 `torch.load`의 기본값은 `weights_only=True`입니다.
예전 YOLOv5/YOLOv7의 전체 모델 객체를 담은 `.pt` 및 일부 데이터 캐시는
이 제한된 로더에서 거부될 수 있습니다. 이 업데이트는 안전장치를 해제하지 않습니다.
재실행 전 직접 학습한 원본 모델인지 확인하고,
[PyTorch 직렬화 안내](https://docs.pytorch.org/docs/2.14/notes/serialization.html)에 따라
신뢰할 수 있는 모델을 `state_dict` 형식으로 변환한 뒤 모델 구조와 함께 불러오거나,
검토한 클래스만 허용해야 합니다. 출처를 모르는 모델에 `weights_only=False`를 적용하지 마세요.
노트북의 `torch.hub.load`는 외부 저장소 코드를 실행하므로 해당 저장소도 신뢰할 수 있어야 합니다.

학습된 가중치와 원본 영상은 저장소에 포함되어 있지 않습니다.
`tests/security_smoke.py`는 다운로드 없이 YOLOv5·YOLOv7·자세 감지 모델의 CPU 추론,
텐서 가중치 저장/복원, 설치 조건 검사 및 이미지 주석 기능을 확인합니다.
기존 가중치로 수행한 추론 결과나 CUDA 학습 성능을 검증하는 테스트는 아닙니다.

`google_app_engine`의 Dockerfile은 Python 3.12용 배포 예제입니다.
각 예제의 `additional_requirements.txt`를 앱의 설치 목록에 합치고 `main:app`을 제공해야 합니다.
