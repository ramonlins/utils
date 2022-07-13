import time

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import pandas as pd
from torchinfo import summary

from models.estimator import Estimator

# turn tensor into numpy array (needed for onnxruntime)
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
def main():
    print("onnx version:", onnx.__version__)
    print("onnxruntime version:", ort.__version__)

    onnx_version = int(onnx.__version__.split('.')[1])
    device = 'cuda'
    batch_size = 1
    input_size = (512, 512)
    total_samples = 10
    is_export = True

    if is_export:
        # deterministic input data
        torch.manual_seed(0)

        model_path = "/home/ramon/Git/adroit/vision_foliage_density/weights/foliage_density_v3/density_model_reg.pth"

        model = Estimator(input_size, model_path).model

        summary(model, input_size=(1, 3, 512, 512))

        # input example necessary to export onnx model
        torch_input_export = torch.randn(batch_size, 3, 512, 512, requires_grad=False).to(device)

        torch.onnx.export(
            model, # model being run
            torch_input_export, # model input (or a tuple for multiple inputs)
            "density.onnx", # where to save the model (can be a file or file-like object)
            verbose=True, # print out a human-readable representation of the model
            opset_version=onnx_version, # the ONNX version to export the model to
            do_constant_folding=True, # constant folding for optimization
            export_params=True, # store the trained parameter weights inside the model file
            input_names=['input'], # the model's input names
            output_names=['output']) # the model's output names
            #dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}) # variable length axes

        # free memory used by the model export
        torch.cuda.empty_cache()

        # print human readable representation of the graph if exist
        try:
            print(onnx.helper.printable_graph(model.graph))
        except AttributeError as error:
            print(error)

    # create a onnx runtime session
    session = ort.InferenceSession("density.onnx")
    if is_profile:
        sess_options = ort.SessionOptions()

        sess_options.enable_profiling = True
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        
        session = ort.InferenceSession("density.onnx", sess_options=sess_options, providers=providers)
    else:
        session = ort.InferenceSession("density.onnx", providers=providers)

    
    # run experiments
    onnx_time_list = []
    for seed in range(10):
        # deterministic input data
        torch.manual_seed(seed)

        # input example necessary to export onnx model
        torch_input = torch.randn(batch_size, 3, 512, 512, requires_grad=False).to(device)
        
        
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
    
if __name__ == '__main__':
    main()