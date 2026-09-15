"""CPU smoke checks: python tests/security_smoke.py (no model downloads)."""
import importlib
import io
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'main/Code'


def check_engine(engine):
    sys.path.insert(0, str(CODE / engine))
    import numpy as np
    import torch
    from PIL import ImageFont
    from models.yolo import Model
    import yaml

    prefix = 'detection_utils' if engine == 'detection' else 'utils'
    general = importlib.import_module(f'{prefix}.general')
    plots = importlib.import_module(f'{prefix}.plots')
    if engine == 'yolov5':
        assert general.check_version('2.13.0', '2.6.0')
        assert not general.check_version('2.5.0', '2.6.0')
        config = CODE / engine / 'models/yolov5n.yaml'
    else:
        # Installed, excluded and inactive requirements must not trigger pip.
        with tempfile.TemporaryDirectory() as directory:
            requirements = Path(directory) / 'requirements.txt'
            requirements.write_text('torch>=2.13.0 # installed\nGitPython>=999\n'
                                    'absent-package; python_version < "3.0"\n')
            with patch('subprocess.check_call') as install:
                general.check_requirements(requirements, exclude=('gitpython',))
                install.assert_not_called()
            with patch('subprocess.check_call') as install:
                general.check_requirements(('torch>=999',))
                install.assert_called_once_with([sys.executable, '-m', 'pip', 'install', 'torch>=999'])
        font = ImageFont.load_default(size=12)
        with patch.object(plots.ImageFont, 'truetype', return_value=font):
            result = plots.plot_one_box_PIL([10, 30, 60, 60], np.zeros((80, 80, 3), dtype=np.uint8),
                                           color=(255, 0, 0), label='person')
        assert result.shape == (80, 80, 3) and result.any()
        labels = [np.array([[0, 0.5, 0.5, 0.2, 0.2]])]
        assert torch.isfinite(general.labels_to_class_weights(labels, nc=1)).all()
        config = CODE / 'yolov7/cfg/deploy/yolov7-tiny.yaml'

    torch.set_num_threads(2)
    if engine == 'detection':
        config = yaml.safe_load(config.read_text())
        config['nc'], config['nkpt'] = 1, 17
        config['head'][-1][2:] = ['IKeypoint', ['nc', 'anchors', 17]]
    else:
        config = str(config)
    model = Model(config, ch=3, nc=1).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    assert torch.isfinite(output[0]).all()
    assert output[0].shape[-1] == (57 if engine == 'detection' else 6)
    if engine == 'detection':
        from Detection import get_pose
        image, poses = get_pose(np.zeros((64, 64, 3), dtype=np.uint8), model, torch.device('cpu'))
        assert image.shape == (1, 3, 960, 960) and np.isfinite(poses).all()

    # Tensor-only checkpoints round-trip under the secure PyTorch loader.
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    # A legacy full-model pickle must not silently disable restricted loading.
    checkpoint = io.BytesIO()
    torch.save(model, checkpoint)
    checkpoint.seek(0)
    try:
        torch.load(checkpoint)
    except pickle.UnpicklingError:
        pass
    else:
        raise AssertionError('The default torch.load accepted an unapproved model class')
    print(f'PASS: {engine} model inference, tensor checkpoint and compatibility checks')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        check_engine(sys.argv[1])
    else:
        for engine in ('yolov5', 'yolov7', 'detection'):
            subprocess.run([sys.executable, __file__, engine], check=True,
                           env={**os.environ, 'YOLO_AUTOINSTALL': 'false', 'YOLOv5_AUTOINSTALL': 'false',
                                'YOLO_CONFIG_DIR': str(Path(tempfile.gettempdir()) / 'objectdetection-smoke')})
