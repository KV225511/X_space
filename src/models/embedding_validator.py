import torch
import numpy as np

def validate_embedding(output):
    assert isinstance(output, torch.Tensor),"Output should be a Tensor Object"
    assert output.ndim == 2, "Output should have 2 dimensions"
    assert output.shape[1]==256,"The shape should haave 256 columns"
    assert output.dtype==torch.float32,"The Datatype should be float32"
    if torch.isnan(output).any():
         raise ValueError("Nan Values are present")
    if torch.isneginf(output).any():
        raise ValueError("Negative Infinite values are present")
         
    if torch.isposinf(output).any():
        raise ValueError("Positive Infinite values are present")
    print("Output format is correct")
    return True
        