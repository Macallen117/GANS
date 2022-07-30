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

        "005": 0,
        "010": 0,
        "015": 0,
        "020": 0,
        "025": 0,
    }

    # path to load and save data in 2D
    real_path = '../data/MA_1D/dataset_real_'
    real_005_path = '../data/MA_1D/dataset_real_005.csv'
    real_010_path = '../data/MA_1D/dataset_real_010.csv'
    real_015_path = '../data/MA_1D/dataset_real_015.csv'
    real_020_path = '../data/MA_1D/dataset_real_020.csv'
    real_025_path = '../data/MA_1D/dataset_real_025.csv'
    real_N_path = '../data/MA_1D/dataset_real_N.csv'

    fake_path = '../data/MA_1D/dataset_fake_'
    fake_005_path = '../data/MA_1D/dataset_fake_005.csv'
    fake_010_path = '../data/MA_1D/dataset_fake_010.csv'
    fake_015_path = '../data/MA_1D/dataset_fake_015.csv'
    fake_020_path = '../data/MA_1D/dataset_fake_020.csv'
    fake_025_path = '../data/MA_1D/dataset_fake_025.csv'
    fake_N_path = '../data/MA_1D/dataset_fake_N.csv'

    real_cycle_path = '../data/MA_1D_CYCLE/dataset_real_'
    real_005_cycle_path = '../data/MA_1D_CYCLE/dataset_real_005.csv'
    real_010_cycle_path = '../data/MA_1D_CYCLE/dataset_real_010.csv'
    real_015_cycle_path = '../data/MA_1D_CYCLE/dataset_real_015.csv'
    real_020_cycle_path = '../data/MA_1D_CYCLE/dataset_real_020.csv'
    real_025_cycle_path = '../data/MA_1D_CYCLE/dataset_real_025.csv'
    real_N_cycle_path = '../data/MA_1D_CYCLE/dataset_real_N.csv'

    fake_cycle_path = '../data/MA_1D_CYCLE/dataset_fake_'
    fake_005_cycle_path = '../data/MA_1D_CYCLE/dataset_fake_005.csv'
    fake_010_cycle_path = '../data/MA_1D_CYCLE/dataset_fake_010.csv'
    fake_015_cycle_path = '../data/MA_1D_CYCLE/dataset_fake_015.csv'
    fake_020_cycle_path = '../data/MA_1D_CYCLE/dataset_fake_020.csv'
    fake_025_cycle_path = '../data/MA_1D_CYCLE/dataset_fake_025.csv'
    fake_N_cycle_path = '../data/MA_1D_CYCLE/dataset_fake_N.csv'


    # path to load and save data in 2D img
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
