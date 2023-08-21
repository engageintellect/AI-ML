import numpy as np
import torch
import torch.nn.functional as F

def numpy_to_python(data):
    if isinstance(data, (np.ndarray, np.generic)):
        return numpy_to_python(data.tolist())
    elif isinstance(data, list):
        return [numpy_to_python(item) for item in data]
    elif isinstance(data, dict):
        return {key: numpy_to_python(value) for key, value in data.items()}
    else:
        return data

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
