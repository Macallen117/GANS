from matplotlib import pyplot as plt

from meter import Meter
from dataset import *
from classification.Lstm_Cnn.model import *
from classification.config import *



if __name__ == '__main__':
    config = Config()
    seed_everything(config.seed)

    # model = RNNModel(1, 64, 'lstm', True).to(config.device)
    # model = RNNAttentionModel(1, 64, 'lstm', False).to(config.device)
    model = CNN(num_classes=3, hid_size=128).to(config.device)

    net_path = 'Lstm_Cnn/net/best_model_epoc23.pth'

    test_dataloader = get_dataloader(phase='test', batch_size=33)
    print(len(test_dataloader))
    data, target = next(iter(test_dataloader))
    data = data.to(config.device)
    target = target.to(config.device)
    print(data.shape, target.shape)

    model.load_state_dict(torch.load(net_path))
    output = model(data).to(config.device)

    meter = Meter()
    meter.init_metrics()

    output = np.argmax(output.detach().cpu().numpy(), axis=1)
    target = np.argmax(target.detach().cpu().numpy(), axis=1)
    meter._compute_cm(output, target)
    confusion_matrix = meter.get_confusion_matrix()

    fig, ax = plt.subplots(figsize=(5, 5))
    cm_ = ax.imshow(confusion_matrix, cmap='hot')
    ax.set_title('Confusion matrix', fontsize=15)
    ax.set_xlabel('Actual', fontsize=13)
    ax.set_ylabel('Predicted', fontsize=13)
    plt.colorbar(cm_)
    plt.show()