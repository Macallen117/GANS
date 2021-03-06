from matplotlib import pyplot as plt
from meter import Meter
from dataset import *
from classification.ResNet.Resnet import *
from classification.config import *


if __name__ == '__main__':
    config = Config()
    seed_everything(config.seed)

    model = resnet18().to(config.device)

    net_path = './ResNet/net/best_model_epoc14.pth'

    test_dataloader = get_imagedataloader(phase='test', batch_size=33)
    print(len(test_dataloader))
    data, target = next(iter(test_dataloader))
    data = data.to(config.device)
    # target = target.to(config.device)
    temp = torch.zeros([len(target), 3]).to(config.device)
    for count, index in enumerate(target):
        temp[count, index] = 1
    target = temp
    print(data.shape, target.shape)

    model.load_state_dict(torch.load(net_path))
    output = model(data)

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