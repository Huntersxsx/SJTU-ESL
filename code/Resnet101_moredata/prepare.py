import torch
from torchvision import transforms, datasets
import os




def load_data(train_data_path, validation_data_path, batch_size, transform: transforms):
    """Load the data from the given folders and return its loaders."""
    train_set = datasets.ImageFolder(root=train_data_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    validation_set = datasets.ImageFolder(validation_data_path, transform=transform)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    training_batches = int(len(train_set) / batch_size)
    validation_batches = int(len(validation_set) / batch_size)

    return train_loader, validation_loader, training_batches, validation_batches


def get_mean_values(train_set: datasets.ImageFolder):
    """ Get mean values from the train set.
    This value will be later used for normalization of the pictures.
    """
    return torch.stack([t.mean(1).mean(1) for t, c in train_set])


def get_transforms():
    """Return transforms for data that will be used in Convolutional network."""
    return transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                               transforms.Normalize([0.153998, 0.153998, 0.153998], [0.308592, 0.308592, 0.308592])])
