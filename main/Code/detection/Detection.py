import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from torchvision import transforms
from detection_utils.datasets import letterbox
from detection_utils.general import non_max_suppression_kpt
from detection_utils.plots import output_to_keypoint
import math

def fall_detection(pose):
    xmin, ymin = (pose[2] - pose[4] / 2), (pose[3] - pose[5] / 2)
    xmax, ymax = (pose[2] + pose[4] / 2), (pose[3] + pose[5] / 2)
    left_shoulder_y = pose[23]
    left_shoulder_x = pose[22]
    right_shoulder_y = pose[26]
    left_body_y = pose[41]
    left_body_x = pose[40]
    right_body_y = pose[44]
    len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))
    left_foot_y = pose[53]
    right_foot_y = pose[56]
    dx = int(xmax) - int(xmin)
    dy = int(ymax) - int(ymin)
    difference = dy - dx
    if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
            len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2) or (
            right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (
            len_factor / 2) and right_shoulder_y > right_body_y - (len_factor / 2)) \
            or difference < 0:
        return True, (xmin, ymin, xmax, ymax)
    return False, None

def prepare_image_for_display(image_tensor):
    _image = image_tensor[0].permute(1, 2, 0) * 255
    _image = _image.cpu().numpy().astype(np.uint8)
    _image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)  
    return _image 

def draw_bounding_boxes(img, model, device):
    
    poses = get_detection_results(img, model, device)
    
    image = letterbox(img, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    
    # _image = prepare_image_for_display(image)
    
    output = []
    if poses is not None:
        for pose in poses:
            is_fall, bbox = fall_detection(pose)
            if is_fall:
                xmin, ymin, xmax, ymax = bbox
                output.append((xmin/960, ymin/960, xmax/960, ymax/960))
                # cv2.rectangle(_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 3)
                # cv2.putText(_image, 'Fallen', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                # print(f"바운딩 박스 좌표: {bbox}")
    
    # plt.imshow(_image)
    # plt.show()
    
    return output if len(output) > 0 else None

def get_detection_model(file_path):
    # 디텍팅 모델 객체를 얻어서 모델 장치 리턴.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = torch.load(file_path, map_location=device)
    model = weights['model']
    _ = model.float().eval()
    if torch.cuda.is_available():
        model = model.half().to(device)
    return model, device

def get_detection_results(image, model, device):
    # 이미지와 모델을 넣으면 디텍팅 결과 (x, y, w, h) 정보 리스트를 리턴함
    image_tensor, poses = get_pose(image, model, device)
    
    if poses is None or len(poses) == 0:
        print("포즈 X")
        return None

    return poses

def get_pose(image, model, device):
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    if torch.cuda.is_available():
        image = image.half().to(device)
    with torch.no_grad():
        output, _ = model(image)
    output = non_max_suppression_kpt(output, 0.7, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    return image, output
