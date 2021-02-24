import numpy as np
import cupy as cp
import argparse
import utility

parser = argparse.ArgumentParser()
parser.add_argument("--filter_size", default=3, type=int)
parser.add_argument("--arr_size", default=100, type=int)
parser.add_argument("--n_images", default=100, type=int)
args = parser.parse_args()


class DummyProgress:
    def update(self):
        pass

    def mark_complete(self):
        pass


dtype = "float32"
cuda = utility.CudaExecuter("float32")

filter_size = args.filter_size
mode = "bright"
progress = DummyProgress()
diff = 0.5
data = np.random.rand(args.n_images, args.arr_size,
                      args.arr_size).astype(dtype)

with cp.cuda.profile():
    cuda.remove_outlier(data, diff, filter_size, mode, progress)
