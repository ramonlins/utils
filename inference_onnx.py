import time

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import pandas as pd

from models.estimator import Estimator

def main():
    print("onnx version:", onnx.__version__)
    print("onnxruntime version:", ort.__version__)

    onnx_version = int(onnx.__version__.split('.')[1])
    device = 'cuda'
    batch_size = 4
    input_size = (512, 512)
    total_samples = 1000
    is_export = False
    onnx_time_list = []

    # run experiments
    for seed in range(100):
        # deterministic input data
        torch.manual_seed(seed)

        # input example necessary to export onnx model
        torch_input = torch.randn(batch_size, 3, 512, 512, requires_grad=False).to(device)
        
        if is_export:
            model_path = "/home/ramon/Git/adroit/vision_foliage_density/weights/foliage_density_v3/density_model_reg.pth"

            model = Estimator(input_size, model_path).model

            torch.onnx.export(
                model, # model being run
                torch_input, # model input (or a tuple for multiple inputs)
                "density.onnx", # where to save the model (can be a file or file-like object)
                opset_version=onnx_version,
                export_params=True, # store the trained parameter weights inside the model file
                input_names=['input'], # the model's input names
                output_names=['output']) # the model's output names
                #dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}) # variable length axes

            # print human readable representation of the graph if exist
            try:
                print(onnx.helper.printable_graph(model.graph))
            except AttributeError as error:
                print(error)

            # free memory used by the model export
            torch.cuda.empty_cache()

        # turn tensor into numpy array (needed for onnxruntime)
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        # create a onnx runtime session
        session = ort.InferenceSession("density.onnx")

        if (total_samples%batch_size) != 0:
            raise Exception("total_samples must be a multiple of batch_size")
        else:
            start = time.time()
            for _ in range(total_samples//batch_size):
                onnx_input = {"input": to_numpy(torch_input)}
                session.run(None, onnx_input)
            end = time.time()

            onnx_time = end - start
    
        onnx_time_list.append(onnx_time)
    
    print(pd.Series(onnx_time_list).describe())
    #print(f"ONNX inference time = {onnx_time}")

if __name__ == '__main__':
    main()