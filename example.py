import torch

from frechet_video_distance import frechet_video_distance


NUMBER_OF_VIDEOS = 16
VIDEO_LENGTH = 15
PATH_TO_MODEL_WEIGHTS = "./pytorch_i3d_model/models/rgb_imagenet.pt"


def main():
    first_set_of_videos = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3, requires_grad=False)
    second_set_of_videos = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3, requires_grad=False) * 255

    fvd = frechet_video_distance(first_set_of_videos, second_set_of_videos, PATH_TO_MODEL_WEIGHTS)
    print("FVD:", fvd.numpy())


if __name__ == "__main__":
    main()
