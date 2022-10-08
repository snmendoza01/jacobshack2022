import json
from flask import Flask 
import torch
from ML_utils import load_weights
from naive_classifier import Net

app = Flask(__name__)
STATE_PATH = "weights.pt"

def find_lbl_idx(img):
    my_net = Net()
    load_weights(my_net, STATE_PATH)
    outputs = my_net(img)
    _, predicted = torch.max(outputs, 1)
    return predicted

@app.route("/predict")
def index(img):
    return json.dumps({"label_idx": find_lbl_idx(torch.randn(img))})