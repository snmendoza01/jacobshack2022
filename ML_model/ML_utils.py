from torch.nn import Module as NeuralNetwork
import torch 

def save_weights(my_model: NeuralNetwork, path: str):
    """Save the weights of an already optimized model
    Args:
        my_model (NeuralNetwork): Model with weights to be saved
        path (str): path to save the model's weights (usually with a .pt extension)
    Returns:
        None
    """
    torch.save(my_model.state_dict(), path)
    return None


def load_weights(new_model: NeuralNetwork, path: str):
    """Load weights into a model
    Args:
        new_model (NeuralNetwork): New model with non-optimized weights
        path (str): Path with the weights to be used
    Returns:
        None: Model is updated internally
    """
    new_model.load_state_dict(torch.load(path))
    return None