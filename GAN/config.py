import random
import numpy as np
import torch


class Config:
    seed = 2021
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    id_to_label = {
        0: "L005",
        1: "L010",
        2: "L015",
        3: "L020",
        4: "L025",
        5: "N",
        6: "R005",
        7: "R010",
        8: "R015",
        9: "R020",
        10: "R025",
    }
    label_to_id = {
        "L005": 0,
        "L010": 1,
        "L015": 2,
        "L020": 3,
        "L025": 4,
        "N": 5,
        "R005": 6,
        "R010": 7,
        "R015": 8,
        "R020": 9,
        "R025": 10,
    }
    id_to_class = {
        0: "L",
        1: "N",
        2: "R",

        0: "L005",
        0: "L010",
        0: "L015",
        0: "L020",
        0: "L025",
        1: "N",
        2: "R005",
        2: "R010",
        2: "R015",
        2: "R020",
        2: "R025",
    }
    class_to_id = {
        "L": 0,
        "N": 1,
        "R": 2,

        "L005": 0,
        "L010": 0,
        "L015": 0,
        "L020": 0,
        "L025": 0,
        "N": 1,
        "R005": 2,
        "R010": 2,
        "R015": 2,
        "R020": 2,
        "R025": 2,
    }
    # path to load real data in 1D
    real_path = '../data/MA_1D/dataset_real.csv'
    # path to save and load generated data
    fake_path = '../data/MA_1D/dataset_fake.csv'

    # path to load real data in 2D img
    test_img_path = '../data/MA_2D/test'
    train_img_path = '../data/MA_2D/train'
    val_img_path = '../data/MA_2D/val'

    # attn_state_path = './net/attn.pth'
    # lstm_state_path = './net/lstm.pth'
    # cnn_state_path = './net/cnn.pth'
    #
    # attn_logs = './log/attn.csv'
    # lstm_logs = './log/lstm.csv'
    # cnn_logs = './log/cnn.csv'

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
        
if __name__ == '__main__':        
    config = Config()
    seed_everything(config.seed)
