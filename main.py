import utils
import yolo
import time

##### 데이터 로드
data_path = "/home/hjkwon/urop-stairs/data/"
rgb_path = data_path + "rgb_data.bin"
depth_path = data_path + "depth_data.bin"
depth_map, rgb_image = utils.align_depth_to_rgb(depth_path, rgb_path, 10)


##### 욜로 디텍션
model = yolo.load_model("/home/hjkwon/urop-stairs/best.pt")
start_time = time.time()
results = yolo.detect(model, rgb_image) #에이서에서 돌릴 때 디텍트 쿠다로 바꾸기

###### 만약 바운딩박스가 여러 개면 가까운거 1개만 남기고 없애기
has_duplicate = utils.check_duplicate(results)
if has_duplicate:
    bbox = utils.remove_extra_box(results,depth_map)
else:
    bbox = tuple(map(int, results[0].boxes.xyxy[0]))

###### 연산량 감소를 위해 roi 크롭, (컬러/뎁스),H,W,채널 형식의 어레이로 반환
rgb_roi, depth_roi = utils.crop_roi(bbox, rgb_image, depth_map)


###### ROI 내에서 탐지된 물체의 height 구하기 (5프레임동안 모아서 평균)





end_time = time.time()
exe_time = end_time-start_time
print(f'⏳{exe_time}초동안 실행')