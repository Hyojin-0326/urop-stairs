import utils
import yolo


##### 데이터 로드
data_path = "/home/hjkwon/urop-stairs/data/"
rgb_path = data_path + "rgb_data.bin"
depth_path = data_path + "depth_data.bin"
depth_map, rgb_image = utils.align_depth_to_rgb(depth_path, rgb_path, 10)


##### 욜로 디텍션
model = yolo.load_model("/home/hjkwon/urop-stairs/best.pt")
results = yolo.detect(model, rgb_path)

###### 만약 바운딩박스가 중복된다면 가까운거 1개만 남기고 없애기


