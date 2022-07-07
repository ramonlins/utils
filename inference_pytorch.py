import time

import torch
import pandas as pd

from models.estimator import Estimator

def main():
    print("pytorch version:", torch.__version__)
    print("cuda version:" , torch.version.cuda)
    print("cudnn version:", torch.backends.cudnn.version())

    # set parameters for the model
    device = 'cuda'
    batch_size = 4
    total_samples = 100
    input_size = (512, 512)
    torch_time_list = []

    # load model
    model_path = "/home/ramon/Git/adroit/vision_foliage_density/weights/foliage_density_v3/density_model_reg.pth"

    predictor = Estimator(input_size, model_path)

    for seed in range(100):
        # deterministic input data
        torch.manual_seed(seed)

        # Create random input data
        torch_inputs = torch.randn(batch_size, 3, 512, 512, requires_grad=False)

        # inference torch
        if (total_samples%batch_size) != 0:
            raise Exception("total_samples must be a multiple of batch_size")
        else:
            samples_size = total_samples//batch_size
            total_start = time.time()
            for _ in range(samples_size):
                torch_input = torch_inputs.to(device)
                predictor.estimate(torch_input)
            total_end = time.time()

            torch_time = total_end - total_start

        torch_time_list.append(torch_time)
    
    print(pd.Series(torch_time_list).describe())
        #print(f"Pytorch total inference time = {torch_time}")

if __name__ == '__main__':
    main()