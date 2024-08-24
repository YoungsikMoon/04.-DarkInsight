import cv2
import math
import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from random import randrange
from glob import glob
import os
from tqdm import tqdm

class AugRotation():
    def __init__(self):
        pass
    
    def warpAffine(self, src, M, dsize, from_bounding_box_only=False):
        return cv2.warpAffine(src, M, dsize)

    def rotate_image(self, image, angle):

        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        image = self.warpAffine(image, M, (nW, nH), False)

        return image

    def crop_to_center(self, old_img, new_img):

        if isinstance(old_img, tuple):
            original_shape = old_img
        else:
            original_shape = old_img.shape
            
        original_width = original_shape[1]
        original_height = original_shape[0]
        original_center_x = original_shape[1] / 2
        original_center_y = original_shape[0] / 2

        new_width = new_img.shape[1]
        new_height = new_img.shape[0]
        new_center_x = new_img.shape[1] / 2
        new_center_y = new_img.shape[0] / 2

        new_left_x = int(max(new_center_x - original_width / 2, 0))
        new_right_x = int(min(new_center_x + original_width / 2, new_width))
        new_top_y = int(max(new_center_y - original_height / 2, 0))
        new_bottom_y = int(min(new_center_y + original_height / 2, new_height))

        canvas = np.zeros(original_shape)

        left_x = int(max(original_center_x - new_width / 2, 0))
        right_x = int(min(original_center_x + new_width / 2, original_width))
        top_y = int(max(original_center_y - new_height / 2, 0))
        bottom_y = int(min(original_center_y + new_height / 2, original_height))

        canvas[top_y:bottom_y, left_x:right_x] = new_img[new_top_y:new_bottom_y, new_left_x:new_right_x]

        return canvas

    def rotate_point(self, origin, point, angle):

        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        
        return qx, qy


    def rotate_annotation(self, origin, annotation, degree):

        new_annotation = copy.deepcopy(annotation)

        angle = math.radians(degree)
        origin_x, origin_y = origin
        origin_y *= -1

        x = annotation["x"]
        y = annotation["y"]
        
        new_x, new_y = map(lambda x: round(x * 2) / 2, self.rotate_point(
            (origin_x, origin_y), (x, -y), angle)
        )

        new_annotation["x"] = new_x
        new_annotation["y"] = -new_y

        width = annotation["width"]
        height = annotation["height"]

        left_x = x - width / 2
        right_x = x + width / 2
        top_y = y - height / 2
        bottom_y = y + height / 2

        c1 = (left_x, top_y)
        c2 = (right_x, top_y)
        c3 = (right_x, bottom_y)
        c4 = (left_x, bottom_y)

        c1 = self.rotate_point(origin, c1, angle)
        c2 = self.rotate_point(origin, c2, angle)
        c3 = self.rotate_point(origin, c3, angle)
        c4 = self.rotate_point(origin, c4, angle)

        x_coords, y_coords = zip(c1, c2, c3, c4)
        new_annotation["width"] = round(max(x_coords) - min(x_coords))
        new_annotation["height"] = round(max(y_coords) - min(y_coords))

        return new_annotation
            
    # 이미지 증강 1장 수행
    def aug_rotate(self, img_path, label_path, degree):
    
        im = np.array(Image.open(img_path), dtype=np.uint8) # 이미지 오픈
        im_w, im_h = im.shape[1], im.shape[0] # 이미지 넓이, 높이

        with open (label_path, "r") as f: # 라벨 파일 오픈
            lst = f.readlines()

        result = [] # 새로운 박스 위치 저장 (중심좌표, w, h)
        
        # 이미지 회전, 크롭
        rotated = self.rotate_image(im, degree)
        rotated = self.crop_to_center(im, rotated) # 넘파이 어레이
        save_img = rotated.copy() # 박스 안그려진 조정된 이미지 리턴용
        
        # 박스 만들기
        for line in lst:
            
            # 라벨 정보 파싱
            line = line.replace("\n", "") 
            line = line.split()
            label = int(line[0])
            cx = int(float(line[1]) * im_w)
            cy = int(float(line[2]) * im_h)
            w = int(float(line[3]) * im_w)
            h = int(float(line[4]) * im_h)
            annotation = {"label":"raccoon","x":cx,"y":cy,"width":w,"height":h}

            # 신규 박스 정보 추출
            origin = (im.shape[1] / 2, im.shape[0] / 2)
            new_annot = self.rotate_annotation(
                origin, annotation, degree
            )
            
            # 신규 박스 넓이, 높이
            width = new_annot["width"] 
            height = new_annot["height"]

            # 신규 박스 중심점
            x = new_annot["x"]
            y = new_annot["y"]
            
            # 신규 박스 xyxy
            xmin = int(new_annot["x"] - (width/2))
            ymin = int(new_annot["y"] - (height/2))
            xmax = int(new_annot["x"] + (width/2))
            ymax = int(new_annot["y"] + (height/2))
            

            cv2.rectangle(rotated, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            cv2.line(rotated, (int(x), int(y)), (int(x), int(y)), (255, 0, 0), 10)
            
            # 센터 좌표 비율 정보로 변경
            x_c = x/im_w
            y_c = y/im_h
            
            # 회전과정에서 이미지 범위를 벗어나는 좌표 처리
            x_c = x_c if x_c >= 0 else 0 
            y_c = y_c if y_c >= 0 else 0
            x_c = x_c if x_c <= 1 else 1
            y_c = y_c if y_c <= 1 else 1

            # 신규 박스 정보 저장
            result.append(f"{label} {x_c} {y_c} {width/im_w} {height/im_h}")
        
        # 이미지 + 신규 박스 출력
        rotated = rotated.astype(np.uint8)
        rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        # plt.imshow(rotated)
        # plt.show()
        
        # 저장용 이미지
        save_img = save_img.astype(np.uint8)
        save_img = cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB)
        
        # 박스그린 이미지, 박스 없는 저장용 이미지, 신규 박스정보 리턴
        return save_img, rotated, result
    
    
    # 원본 이미지 저장
    def save_img(self, img_save_path, img):
        cv2.imwrite(img_save_path, img)
        
    # 라벨 정보 저장
    def save_label(self, label_save_path, result):
        
        with open (label_save_path, "w") as f:
            for i, line in enumerate(result):
                # 마지막줄은 엔터를 빼주자
                if i == len(result) - 1:
                    f.write(line)
                else:   
                    f.write(line+'\n')
    
    
    # 증강 여러장 수행 및 이미지파일, 라벨파일 저장 (랜덤 로테이션)
    def run_augment(self, extension, img_root_folder, label_root_folder, 
                    aug_img_save_folder, aug_boximg_save_folder, aug_label_save_folder, count):
        
        img_path_lst = glob(img_root_folder + "\\*." + extension)
        label_path_lst = glob(label_root_folder + "\\*.txt")
        
        for img_path, label_path in zip(tqdm(img_path_lst), label_path_lst):
            
            # count 갯수만큼 회전 증강
            for i in range(1, count+1):
                
                # 증강 이미지, 라벨 파일이름 및 경로 설정
                aug_img_filename = img_path.split("\\")[-1].split(f".{extension}")[0] + f"_aug{i}." + extension 
                aug_boximg_filename = img_path.split("\\")[-1].split(f".{extension}")[0] + f"_aug{i}_box." + extension 
                aug_label_filename = label_path.split("\\")[-1].split(".txt")[0] + f"_aug{i}.txt"
                aug_img_save_path = os.path.join(aug_img_save_folder, aug_img_filename)
                aug_boximg_save_path = os.path.join(aug_boximg_save_folder, aug_boximg_filename)
                aug_label_save_path = os.path.join(aug_label_save_folder, aug_label_filename)

                # 회전 각도 설정
                degree = randrange(30,330) 
                
                # 회전 수행
                img, boximg, result = self.aug_rotate(img_path, label_path, degree)                
                
                # 이미지 및 라벨 저장
                self.save_img(aug_img_save_path, img)
                self.save_img(aug_boximg_save_path, boximg)
                self.save_label(aug_label_save_path, result)
        
        return "완료"
    
    # 증강 여러장 수행 및 이미지파일, 라벨파일 저장 (70~-90도 지정 로테이션)
    def run_specific_augment(self, extension, img_root_folder, label_root_folder, 
                    aug_img_save_folder, aug_boximg_save_folder, aug_label_save_folder, count):
        
        img_path_lst = glob(img_root_folder + "\\*." + extension)
        label_path_lst = glob(label_root_folder + "\\*.txt")
        
        for img_path, label_path in zip(tqdm(img_path_lst), label_path_lst):
            
            
            # count 갯수만큼 회전 증강
            # 기존거에서 이어서 하느라 3 더해서 해줌. 원래는 0부터
            # 기존거에서 이어서 하느라 3 다시 더해서 해줌. 원래는 0부터
            for i in range(1+3+3, 3+3+count+1):
                
                # 회전 각도 설정
                # degree = randrange(50,90)
                
                # 추가로 3장 더 
                degree = randrange(70,95) 
                 
                # 증강 이미지, 라벨 파일이름 및 경로 설정
                aug_img_filename = img_path.split("\\")[-1].split(f".{extension}")[0] + f"_s_aug{i}." + extension 
                aug_boximg_filename = img_path.split("\\")[-1].split(f".{extension}")[0] + f"_s_aug{i}_box." + extension 
                aug_label_filename = label_path.split("\\")[-1].split(".txt")[0] + f"_s_aug{i}.txt"
                aug_img_save_path = os.path.join(aug_img_save_folder, aug_img_filename)
                aug_boximg_save_path = os.path.join(aug_boximg_save_folder, aug_boximg_filename)
                aug_label_save_path = os.path.join(aug_label_save_folder, aug_label_filename)

                
                # 회전 수행
                img, boximg, result = self.aug_rotate(img_path, label_path, degree)                
                
                # 이미지 및 라벨 저장
                self.save_img(aug_img_save_path, img)
                self.save_img(aug_boximg_save_path, boximg)
                self.save_label(aug_label_save_path, result)
        
        return "완료"
    
    # 이미지 + 박스 출력 (라벨 : label + xywh 비율)
    def view_box_image(self, img_path, label_path, sampling=False):
        
        img = cv2.imread(img_path)
        img_w = img.shape[1]
        img_h = img.shape[0]
        
        # 라벨 파싱 및 박스 그리기
        with open (label_path, "r") as f:
            box_lst = f.readlines()
        
        for box in box_lst:
            
            info = box.split()
            
            # 중심좌표 및 w, h
            label = int(info[0]) 
            cx = float(info[1]) * img_w
            cy = float(info[2]) * img_h
            w = float(info[3]) * img_w
            h = float(info[4]) * img_h
            
            # xyxy 좌표
            xmin = int(cx - w/2)
            ymin = int(cy - h/2)
            xmax = int(cx + w/2)
            ymax = int(cy + h/2)
            
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 4)
        
        # isSampling일 경우 이미지만 반환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if sampling:
            return img
        
        plt.axis("off")
        plt.imshow(img)


    
    # 이미지 여러 개 출력 (라벨 : label + xywh 비율)
    def view_sample_box_image(self, img_folder, label_folder, count):
        
        img_lst = glob(img_folder + "\\*.jpg")
        label_lst = glob(label_folder + "\\*.txt")
        
        col = 2
        row = count//col + 1 if count % col > 0 else count//col
        fig = plt.figure(figsize = (8,8))
        
        for i in range(count):
            idx = randrange(len(img_lst))
            img = self.view_box_image(img_lst[idx], label_lst[idx], True)
            fig.add_subplot(row, col, i+1)
            fig.tight_layout()
            plt.title(img_lst[idx].split("\\")[-1], fontsize=8)
            plt.axis("off")
            plt.imshow(img)
        
        plt.show()