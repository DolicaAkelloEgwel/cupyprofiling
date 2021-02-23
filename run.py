import numpy as np
import cupy as cp
import utility

class DummyProgress:
    def update(self):
        pass
    def mark_complete(self):
        pass

dtype = "float32"
cuda = utility.CudaExecuter("float32")

filter_size = 3
mode = "bright"
progress = DummyProgress()
diff = 0.5
data = np.random.rand(20,500,500).astype(dtype)

with cp.cuda.profile():
    cuda.remove_outlier(data, diff, filter_size, mode, progress)
