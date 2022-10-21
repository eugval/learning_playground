import torch
import numpy as np



def min_max_stardardisation_to_01(image):
    image_min = np.min(image)
    image_max = np.max(image)
    range = (image_max-image_min)

    return (image-image_min)/range




def normalise_rgb( data_array, segmentation_masks=None, return_stats=False, mu=None, std=None):
    """

    :param data_array:Assumes a pytorch tensor
    :return:
    """

    if (mu is not None and std is not None):
        pass
    elif (segmentation_masks is not None):
        # TODO : check data array is not overriden
        # Check broadcasting of mu is correct

        normalisation_constant = torch.sum(segmentation_masks, dim=(2, 3), keepdim=True)
        mu = torch.sum(data_array * segmentation_masks, dim=(2, 3), keepdim=True) / normalisation_constant
        std = torch.sum(torch.sqrt(torch.square((data_array - mu)) * segmentation_masks), dim=(2, 3),
                        keepdim=True) / normalisation_constant

    else:
        mu = data_array.mean(dim=(2, 3), keepdim=True)
        std = data_array.std(dim=(2, 3), keepdim=True)

    if (return_stats):
        return (data_array - mu) / (std + 1e-6), mu, std
    else:
        return (data_array - mu) / (std + 1e-6)



def train_val_spilt(data_grabber, ratio, seed):
    dataset_len = len(data_grabber)
    training_dataset_len = int(ratio * dataset_len)

    training_dataset, validation_dataset = torch.utils.data.random_split(data_grabber,
                                                                         [training_dataset_len,
                                                                          dataset_len - training_dataset_len],
                                                                         generator=torch.Generator().manual_seed(seed))

    return training_dataset, validation_dataset
