import re
import torch
import torchvision
import rasterio as rio
import numpy as np
from tqdm import tqdm
from torch.utils import data
from constants import classes_to_int

# Create normalizer for 13 bands with precomputed means and standard deviations across all bands
means_tuple = (
    1353.7269257269966,
    1117.2022923538773,
    1041.8847248444733,
    946.5542548737702,
    1199.1886644965277,
    2003.0067999222367,
    2374.008444688585,
    2301.2204385489003,
    732.1819500777633,
    1820.6963775318286,
    1118.2027229275175,
    2599.7829373281975,
)
stds_tuple = (
    65.29657739037496,
    153.77375864458085,
    187.69931299271406,
    278.1246366855392,
    227.92409611864002,
    355.9331571735718,
    455.13290021052626,
    530.7795614455541,
    98.92998227431653,
    378.16138952053035,
    303.10651348740964,
    502.16376466306053
)
train_normalizer = torchvision.transforms.Normalize(means_tuple, stds_tuple)

submission_means_tuple = (
    380.17328711583616,
    400.1497676971955,
    628.8646132355601,
    578.870857455104,
    943.4272711885449,
    1826.2433534560898,
    2116.6662455740857,
    2205.972884006897,
    2266.934157142567,
    1487.6910683644517,
    959.236167229867,
    2281.1860589241937
)
submission_stds_tuple = (
    115.17434877174112,
    209.14842754591166,
    241.20653977105658,
    301.1056228200069,
    269.5139533673432,
    420.2497496130561,
    503.8183661547185,
    598.040304209199,
    403.93781724898935,
    398.143166872752,
    342.44079144555366,
    529.4133153492427
)
submission_normalizer = torchvision.transforms.Normalize(
    submission_means_tuple, submission_stds_tuple
)


# Define Dataset
class CustomDataset(data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Extract bands
        with rio.open(sample, "r") as d:
            img = d.read([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        tens = torch.tensor(img.astype(int))

        # Normalize
        tens = train_normalizer(tens.float())

        # Extract label
        label = sample.split("/")[-1].split("_")[0]
        label_id = classes_to_int[label]

        return tens, label_id


# Alternatively: In-memory dataset
class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, samples, normalizer=train_normalizer):
        self.x = []
        self.y = []
        for sample in tqdm(samples, desc="Loading training samples"):
            # Extract bands
            with rio.open(sample, "r") as d:
                img = d.read([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13])
            tens = torch.tensor(img.astype(int))

            # Normalize
            tens = normalizer(tens.float())

            # Extract label
            label = sample.split("/")[-1].split("_")[0]
            label_id = classes_to_int[label]
            self.x.append(tens)
            self.y.append(label_id)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class SubmissionDataset(torch.utils.data.Dataset):
    def __init__(self, submission_samples, normalizer=submission_normalizer):
        self.x = []
        for _, submission_sample in tqdm(
            sorted(
                {
                    # Sort files by index
                    int(re.findall("\d+", submission_sample)[0]): submission_sample
                    for submission_sample in submission_samples
                }.items()
            ),
            desc="Loading submission samples"
        ):
            # Extract bands
            img = np.load(submission_sample)

            # SWAP BANDS
            tmp = img[:, :, 8].copy()
            img = np.delete(img, 8, axis=2)
            img = np.insert(img, 11, tmp, axis=2)

            tens = torch.from_numpy(img.astype(int))
            tens = tens.permute(2, 1, 0)

            # Normalize
            tens = normalizer(tens.float())
            self.x.append(tens)

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)
