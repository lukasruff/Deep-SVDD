from datasets.__local__ import implemented_datasets
from datasets.mnist import MNIST_DataLoader
from datasets.cifar10 import CIFAR_10_DataLoader
from datasets.GTSRB import GTSRB_DataLoader


def load_dataset(learner, dataset_name, pretrain=False):

    assert dataset_name in implemented_datasets

    if dataset_name == "mnist":
        data_loader = MNIST_DataLoader

    if dataset_name == "cifar10":
        data_loader = CIFAR_10_DataLoader

    if dataset_name == "gtsrb":
        data_loader = GTSRB_DataLoader

    # load data with data loader
    learner.load_data(data_loader=data_loader, pretrain=pretrain)

    # check all parameters have been attributed
    learner.data.check_all()
