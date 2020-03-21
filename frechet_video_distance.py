import numpy as np
from scipy.linalg import sqrtm
import torch
from torch.nn.functional import interpolate

from pytorch_i3d_model.pytorch_i3d import InceptionI3d


def preprocess(videos, target_resolution):
    # n, t, h, w, c -> n, c, t, h, w
    reshaped_videos = videos.permute(0, 4, 1, 2, 3)
    size = [reshaped_videos.size()[2]] + list(target_resolution)
    resized_videos = interpolate(reshaped_videos, size=size, mode='trilinear', align_corners=False)
    scaled_videos = 2 * resized_videos / 255. - 1
    return scaled_videos


def get_statistics(activations):
    mean = torch.mean(activations, 0)
    bias = (activations - mean).reshape(activations.size()[0], activations.size()[1], 1)
    cov = bias.matmul(bias.transpose(1, 2)).mean(0) * bias.size()[0] / (bias.size()[0] - 1)
    return mean, cov


def calculate_fvd_from_activations(first_activations, second_activations):
    eps = 1e-20
    f_mu, f_cov = get_statistics(first_activations)
    s_mu, s_cov = get_statistics(second_activations)

    mu_dist = torch.norm(f_mu - s_mu).pow(2)
    matrix = f_cov.matmul(s_cov).detach().numpy()
    sqrt_matrix = torch.from_numpy(sqrtm(matrix + eps * np.eye(matrix.shape[0])).real).type_as(f_cov)
    cov_dist = 2. * torch.trace(f_cov + s_cov - 2. * sqrt_matrix)
    return mu_dist + cov_dist


def frechet_video_distance(first_set_of_videos, second_set_of_videos, path_to_model_weights):
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load(path_to_model_weights))
    i3d.train(False)

    print("Calculating activations for the first set of videos...")
    first_activations = i3d(preprocess(first_set_of_videos, (224, 224))).squeeze()
    print("Calculating activations for the second set of videos...")
    second_activations = i3d(preprocess(second_set_of_videos, (224, 224))).squeeze()
    return calculate_fvd_from_activations(first_activations, second_activations)
