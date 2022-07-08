import time

import torch
import pandas as pd
from torchinfo import summary

from models.estimator import Estimator

def main():
    print("pytorch version:", torch.__version__)
    print("cuda version:" , torch.version.cuda)
    print("cudnn version:", torch.backends.cudnn.version())

    # set parameters for the model
    device = 'cuda'
    batch_size = 1
    total_samples = 10
    input_size = (512, 512)
    is_profile = False

    # load model
    model_path = "/home/ramon/Git/adroit/vision_foliage_density/weights/foliage_density_v3/density_model_reg.pth"

    predictor = Estimator(input_size, model_path)
    
    summary(predictor.model, input_size=(batch_size, 3, 512, 512))

    torch_time_list = []
    for seed in range(10):
        # deterministic input data
        torch.manual_seed(seed)

        # Create random input data
        torch_inputs = torch.randn(batch_size, 3, 512, 512, requires_grad=False)

        # inference torch
        if (total_samples%batch_size) != 0:
            raise Exception("total_samples must be a multiple of batch_size")
        else:
            samples_size = total_samples//batch_size
            if is_profile:
                with torch.autograd.profiler.profile(use_cuda=True) as onnx_profiler:    
                    for _ in range(samples_size):
                        torch_input = torch_inputs.to(device)
                        predictor.estimate(torch_input)
            else:
                total_start = time.time()
                for _ in range(samples_size):
                    torch_input = torch_inputs.to(device)
                    predictor.estimate(torch_input)
                total_end = time.time()

                torch_time = total_end - total_start
                torch_time_list.append(torch_time)
    
    if is_profile:
        print(onnx_profiler.key_averages().table(sort_by="self_cuda_time_total")) 
    else:
        print(pd.Series(torch_time_list).describe())
    
if __name__ == '__main__':
    main()