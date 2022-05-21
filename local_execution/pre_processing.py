import torch
import torchvision


def create_normalizer_for_dataset(dataset, verbose: bool = False) -> torchvision.transforms.transforms.Normalize:
    """
    Method returning a normalizer which sets mean to 0 and std to 1 for dataset
    :param dataset: Dataset to compute statistics for the normalizer from
    :param verbose: set True if you want to print the mean and std vectors
    :return: normalizer
    """
    # ONLY EXECUTE IF NEEDED: Compute means and Standard deviation for all bands across all images
    band_means = {}
    band_stds = {}
    # Data needs to be not normalized for this computation
    for x in dataset.x:
        means = torch.mean(x.float(), dim=(1, 2))
        stds = torch.std(x.float(), dim=(1, 2))

        for i, mean in enumerate(means):
            band_means[i] = band_means.get(i, 0) + float(mean)

        for i, std in enumerate(stds):
            band_stds[i] = band_stds.get(i, 0) + float(std)

    means_tuple = tuple()
    for value in band_means.values():
        means_tuple += (value / len(dataset.x),)

    stds_tuple = tuple()
    for value in band_stds.values():
        stds_tuple += (value / len(dataset.x),)
    if verbose:
        print(means_tuple)
        print(stds_tuple)

    normalizer = torchvision.transforms.Normalize(means_tuple, stds_tuple)
    return normalizer
