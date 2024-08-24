import sys
# detection 패키지 등록
if 'detection' not in sys.path:
    sys.path.append('detection')
    
import cv2
import matplotlib.pyplot as plt
from IPython import display
from random import randint
from math import sqrt
import torch
import traceback
import detection.Detection as dt
from detection.models import common, yolo
import importlib
importlib.reload(dt)

class Tracking_Both:
    
    # model : Yolo 모델
    def __init__(self, model, pt_file_path, isUltra=False):
        self.model = model
        
        # 내부적으로 torch.hub.load()시 모듈 파일을 강제 지정
        # 다른 모델을 불러올 때 앞서 만들어진 모듈이름을 사용하면 에러가 나기 때문에
        # 다시 포즈 모델에 맞는 모듈 파일로 매칭시켜줘야함
        sys.modules["models.common"] = sys.modules["detection.models.common"]
        sys.modules["models.yolo"] = sys.modules["detection.models.yolo"]
        self.pose_model, self.device = dt.get_detection_model(pt_file_path)
        
        self.isUltra = isUltra
    
    # 좌표 간 거리 측정
    # param
        # 좌표1 (frame_num, x, y)(tuple)
        # 좌표2 (x),(y) (int)
    # return : 거리        
    def dist(self, p, x, y):
        return sqrt((p[1]-x)**2 + (p[2]-y)**2)


    # 동영상 좌표 추적
    # param
        # obj_thres : 동일 객체 인식할 범위
        # update_thres : 객체 삭제 임계값 프레임 수 (해당 횟수만큼 업데이트 안되면 삭제)
        # tracking_count : 좌표 추적 프레임 횟수
        # 동영상 파일 주소 (str)
        # 앞 프레임 수 100프레임으로 제한 (bool)
        # 영상 저장 주소 (str, 옵션 없으면 저장 안함)
    # return
        # 객체별 추적 좌표(ps_route : dict)
        # 객체별 이동거리 정보(ps_dist : dict) -> 테스트 저장용
        # 객체별 전체 추적 좌표(ps_all_route : dict) -> 테스트 저장용
        # 인식된 프레임 수 (recognized_count : int) -> 모델선정용
    def track_video(self, obj_thres, update_thres, tracking_count, video_path, isSample=False, save_path=""):
        
        video = cv2.VideoCapture(video_path) # 동영상 오픈
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if len(save_path) > 0:
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))          
        
        ps_route = dict() # 경로 저장 {객체번호 : [업데이트, 컬러, (중심좌표), (중심좌표)...]}
        ps_dist = dict() # 거리 정보 저장 {객체번호 : [(w, h, dist)]}
        ps_all_route = dict() # 안지우고 모든 경로 정보 저장 (테스트용)
        
        ps_num = 0 # 객체 번호
        
        try: # 중간에 중단되도 메모리 해제하기 위함
            # 한 프레임씩 작업
            sample_count = 0
            
            frame_num = 0 # 프레임 수
            recognized_count = 0 # 인식된 프레임 수 (얼굴이라는 보장은 없음..)
            
            isSpeedOver = dict() # 속도 임계값 넘어선 여부
            isPoseDetected = [False, 15] # 눕는 포즈 감지 여부
            
            while True:
                
                # 비디오 읽기 (RGB순으로 변환)
                ret, img = video.read()
                if ret == False:
                    break
                
                
                # 샘플 설정할 경우 100프레임만 돌고 스톱
                if isSample:
                    sample_count += 1
                    if sample_count == 100:
                        break
                
                # 헤드 모델 통과 
                result = None
                if self.isUltra:
                    with torch.no_grad():
                        result = self.model.predict(img, conf=0.7)
                else:
                    # v5,v7는 RGB로 바꿔줘야함
                    # v8,v10은 BGR로 그대로 넣어줘야함
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    with torch.no_grad():
                      result = self.model(img)

                # 포즈 감지 모델 통과
                # 함수이름 귀찮아서 안바꿈
                with torch.no_grad():
                    pose_result = dt.draw_bounding_boxes(img, self.pose_model, self.device)
                
                # 일단 박스 하나만. xyxy 비율
                pose_ratio = False if not pose_result else pose_result[0]                
                
                # 필요한 클래스 좌표만 불러오기
                p_class = 0
                p_center, p_xyxy = None, None
                if self.isUltra:
                    cls = result[0].boxes.cls
                    p_center = result[0].boxes.xywh
                    p_center = p_center[cls==p_class].type(torch.int).tolist()
                    p_xyxy = result[0].boxes.xyxy
                    p_xyxy = p_xyxy[cls==p_class].type(torch.int).tolist()
                    
                else:
                    p_center = result.xywh[0].type(torch.int) # 모든 객체 좌표정보 (중심값, 가로, 세로)
                    p_center = p_center[p_center[:, 5] == p_class].tolist() # 원하는 클래스 좌표정보
                    p_xyxy = result.xyxy[0].type(torch.int) # 모든 객체 좌표정보 (xmin, ymin, xmax, ymax)
                    p_xyxy = p_xyxy[p_xyxy[:, 5] == p_class].tolist() # 원하는 클래스 좌표정보
                    # p_conf_lst = result.pred[0] # (xyxy, conf, cls) 리스트

                # 좌표가 특정 갯수 넘어가면 젤 처음꺼 하나 지우기
                # 업데이트가 기준횟수만큼 안된 경우에는 객체 지우기
                del_idx = []
                for n, d_lst in ps_route.items():
                    
                    d_lst[0] -= 1 # 한 프레임마다 업데이트 정보 하나씩 디스카운트
                    
                    if len(d_lst) > tracking_count: # 갯수 초과하면 처음꺼 삭제
                        del d_lst[3] # 좌표는 3번부터 시작
                    
                    if d_lst[0] == 0: # 업데이트정보가 0이 됐을 경우
                        del_idx.append(n)
                
                # 업데이트 횟수 지난 객체 삭제
                for n in del_idx: 
                    del ps_route[n]
                    del isSpeedOver[n]
                
                # 객체별 사각형 및 중심점에 점 찍기
                img_shape_min = min(img.shape[:2]) # 이미지 크기 min값
                for i in range(len(p_center)):
                    p_c, p_xy = p_center[i], p_xyxy[i] # 중심점 좌표, xyxy좌표
                    # p_conf = round(p_conf_lst[i][4].item(), 5) # 컨피던스값
                    
                    # 이전 프레임에서 가장 가까운 좌표를 가진 객체 선별
                    min_ps = None # 가장 가까운 객체 번호
                    min_dist = img_shape_min # 가장 가까운 거리 저장용 (한 객체마다 초기화)
                    
                    if len(ps_route) > 0: # 저장된 객체가 하나라도 있을 때
                        
                        for n, d_lst in ps_route.items():
                            d = self.dist(d_lst[len(d_lst)-1], p_c[0], p_c[1]) # 해당 객체와 마지팍 좌표와의 거리계산
                            if d < min_dist: # 가장 가까운 거리일 경우
                                min_dist = d # 가까운 거리 업데이트
                                min_ps = n # 가까운 객체번호 업데이트
                    
                    if min_dist <= obj_thres and len(ps_route) > 0: # 가장 가까운 거리가 기준점 안에 있을 경우 
                        
                        # 인식안된 구간 프레임 좌표값 채워주기
                        before_info = ps_route[min_ps][len(ps_route[min_ps])-1] # 직전 좌표값
                        before_info_all = ps_all_route[min_ps][len(ps_all_route[min_ps])-1] # 직전 로그값(xywh)
                        interval = frame_num - before_info[0] # 직전 좌표값의 프레임과의 차이
                        if interval > 1: # 직전 프레임과 차이가 1보다 클 경우 (중간이 비었다는 뜻)
                            
                            # 비어있는 프레임 사이 1프레임당 좌표 거리 계산 
                            x_distance = int((p_c[0] - before_info[1]) / (interval - 1))
                            y_distance = int((p_c[1] - before_info[2]) / (interval - 1))
                            
                            # 비어있는 프레임 사이 1프레임당 w, h 크기 계산
                            w_distance = int((p_c[2] - before_info_all[3]) / (interval - 1))
                            h_distance = int((p_c[3] - before_info_all[4]) / (interval - 1))
                            
                            # 채워야 할 갯수만큼 좌표 추가
                            for i in range(interval - 1):
                                
                                # 중간 값을 채워줌 (프레임, x좌표, y좌표)
                                hypo_frame = before_info[0]+i+1
                                hypo_x = before_info[1]+((i+1)*x_distance)
                                hypo_y = before_info[2]+((i+1)*y_distance)
                                ps_route[min_ps].append((hypo_frame, hypo_x, hypo_y))
                                
                                # 로그도 남겨줌
                                hypo_w = before_info_all[3]+((i+1)*w_distance)
                                hypo_h = before_info_all[4]+((i+1)*h_distance)
                                ps_all_route[min_ps].append((hypo_frame, hypo_x, hypo_y, hypo_w, hypo_h))
                        
                        ps_route[min_ps].append((frame_num, p_c[0], p_c[1])) # 해당 객체에 좌표 추가
                        ps_route[min_ps][0] = update_thres # 업데이트 디스카운트 다시 시작
                        
                        # 전체 경로 저장용
                        ps_all_route[min_ps].append((frame_num, p_c[0], p_c[1], p_c[2], p_c[3])) # 중심 x, y, w, h 
                        
                        # 이전 프레임 대비 이동거리정보 추가
                        # 현재 얼굴크기(w-p_c[2])로 나눠서 정규화
                        ps_dist[min_ps].append((p_c[0], p_c[1], min_dist))
                        
                        # Dist Threshold(1) 넘어가면 상태값 변경
                        # 0,1은 다른정보, 2번 첫 좌표는 0이므로 패스. 3번-8번부터 거리계산시작
                        if len(ps_route[min_ps]) > 7:
                            
                            # 5프레임 이전 좌표정보
                            before_xy_5frame = ps_route[min_ps][len(ps_route[min_ps])-6]
                            
                            # 5프레임 이전 좌표정보와의 거리
                            dist_5frame = self.dist(before_xy_5frame, p_c[0], p_c[1])
                            
                            # 5프레임 이전 좌표정보와의 정규화 거리가 임계값 초과할 경우 상태값 변경
                            if dist_5frame/p_c[2] > 1.1:
                                isSpeedOver[min_ps][0] = True
                        
                        
                    else: # 기준점 밖에 있을 경우 또는 가장 처음일 경우 새로운 객체만들고 좌표 추가
                        
                        # idx0 : 업데이트 안될때마다 차감 -> 0이 되면 화면 밖으로 나간 것으로 간주, 객체 삭제
                        # idx1 : 해당 객체 컬러정보
                        ps_route[ps_num] = [update_thres, (randint(0,255),randint(0,255),randint(0,255)), (frame_num, p_c[0], p_c[1])]
                        
                        # 전체 경로 저장용
                        ps_all_route[ps_num] = [(frame_num, p_c[0], p_c[1], p_c[2], p_c[3])] # 중심 x, y, w, h
                        
                        # 이동거리정보 추가
                        ps_dist[ps_num] = [(p_c[0], p_c[1], 0)]
                        
                        # 속도 임계점 여부 등록 (디폴트 False)
                        isSpeedOver[ps_num] = [False]
                        
                        # 가장 가까운 사람은 자기 자신
                        min_ps = ps_num
                        
                        # 다음 객체번호로 바꿔줌
                        ps_num += 1 
                    
                    
                    # 사각형 그리기
                    # 쓰러짐으로 감지됐을 경우                    
                    if min_ps in isSpeedOver.keys() and isSpeedOver[min_ps][0]:
                        
                        # 쓰러짐 사각형
                        cv2.rectangle(img, (p_xy[0], p_xy[1]), (p_xy[2], p_xy[3]), ps_route[min_ps][1], 10)
                        
                        # 쓰러짐 경보 출력
                        cv2.putText(img, f"!!Speed Alarm!!", (p_xy[0], p_xy[1]-40), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 3, cv2.LINE_AA) 
                    else:
                        cv2.rectangle(img, (p_xy[0], p_xy[1]), (p_xy[2], p_xy[3]), (255,0,0), 2)
                        # cv2.putText(img, str(p_conf), (p_xy[0], p_xy[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 1, cv2.LINE_AA) #컨피던스 출력
                    
                    if isSpeedOver[min_ps][0] and isPoseDetected[0]:
                        cv2.putText(img, f"!!Fall Detection!!", (200, 200), cv2.FONT_HERSHEY_PLAIN, 8, (0,0,255), 3, cv2.LINE_AA) 

                
                # 객체 인식된 프레임 수 카운트 (모델 성능 확인용)
                if len(p_center) > 0:
                    recognized_count += 1 

                # 모든 이동경로 중심좌표에 점 찍기
                for n, d_lst in ps_route.items():
                    
                    color = d_lst[1]
                    for i, lst in enumerate(d_lst):
                        if i > 2: # 좌표는 3부터 시작
                            cv2.line(img, (lst[1], lst[2]), (lst[1], lst[2]), color, 10)
                            cv2.putText(img, str(n), (lst[1], lst[2]), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 1, cv2.LINE_AA)
                
                # 포즈 감지 시 사각형 그리기
                # 원래 머리랑 동일 객체 인식 후 하나씩 작업해야하는데 시간 없어서 그냥 한 사람만 있다고 가정하고 작성함..
                if pose_ratio:
                    if not isPoseDetected[0]:
                        isPoseDetected[0] = True
                        
                    isPoseDetected[1] = 15    
                    pose_xmin = int(pose_ratio[0] * width)
                    pose_ymin = int(pose_ratio[1] * height)
                    pose_xmax = int(pose_ratio[2] * width)
                    pose_ymax = int(pose_ratio[3] * height)
                    cv2.rectangle(img, (pose_xmin, pose_ymin), (pose_xmax, pose_ymax), (0,0,255), 10)
                    cv2.putText(img, f"!!Pose Alarm!!", (pose_xmin, pose_ymax + 100), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 3, cv2.LINE_AA) 

                if isPoseDetected[0] and not pose_ratio:
                    isPoseDetected[1] -= 1
                    
                    if isPoseDetected[1] == 0:
                        isPoseDetected[0] = False
                        for n, v in isSpeedOver.items():
                            v[0] = False
                

                # # 이미지 출력
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                display.clear_output(wait=True)
                plt.imshow(img)  
                plt.show()
                
                # 비디오 저장
                if len(save_path) > 0:
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR로 변환
                    writer.write(img)
                    
                frame_num += 1 # 프레임 수
    
        except(Exception) as e:
            print(traceback.format_exc())
        
        finally:
            # 메모리 해제
            video.release()
            if writer is not None:
                writer.release()
            
        return ps_route, ps_dist, ps_all_route, recognized_count   
    
    
    
    # 사진 경로 주면 박스쳐서 보여주기
    # param
        # 모델 (model)
        # 이미지 경로 (str)
    # return : CV2 형식 BGR 이미지 파일
    def track_image(self, img_path):

        img = cv2.imread(img_path) # 이미지 오픈
        result = self.model(img) # 모델 통과

        class_num = 0 # 검색할 클래스 넘버
        xyxy = result.xyxy[0] # xyxy좌표 (비율 아님)
        xyxy = xyxy[xyxy[:, 5] == class_num]

        xywh = result.xywh[0] # xywh좌표 (비율 아님)
        xywh = xywh[xywh[:, 5] == class_num]

        for center_info, box_info in zip(xywh, xyxy):
            
            # 중심좌표
            x_c = int(center_info[0])
            y_c = int(center_info[1])
            
            # 박스좌표
            xmin = int(box_info[0])
            ymin = int(box_info[1])
            xmax = int(box_info[2])
            ymax = int(box_info[3])
            
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            cv2.line(img, (x_c, y_c), (x_c, y_c), (0,0,255), 5)

        img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display.clear_output(wait=True)
        plt.imshow(img_cvt)
        plt.show()
        
        # 리턴은 cv2형식 BGR로
        return img