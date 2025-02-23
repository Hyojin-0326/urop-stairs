import pycuda.driver as drv
import pycuda.autoinit
print(drv.get_version())  # CUDA 버전 출력
print("done")