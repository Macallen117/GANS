import random
import numpy as np
import torch


class Config:
    seed = 2021
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    id_to_label = {
        0: "005",
        1: "010",
        2: "015",
        3: "020",
        4: "025",
        5: "N",
    }
    label_to_id = {
        "005": 0,
        "010": 1,
        "015": 2,
        "020": 3,
        "025": 4,
        "N": 5,
    }
    id_to_class = {
        0: "L",
        1: "N",
    }
    class_to_id = {
        "L": 0,
        "N": 1,
    }
    # path to load real data in 1D
    real_path = '../data/MA_1D/dataset_real.csv'
    # path to save and load generated data
    fake_path = '../data/MA_1D/dataset_fake.csv'

    # path to load real data in 2D img
    test_img_path = '../data/MA_2D/test'
    train_img_path = '../data/MA_2D/train'
    val_img_path = '../data/MA_2D/val'

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
        
if __name__ == '__main__':        
    config = Config()
    seed_everything(config.seed)
