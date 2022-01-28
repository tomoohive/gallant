import sys
import os
import glob


def _get_dataset_paths(directory):
    files = os.listdir(directory)
    file_paths = []
    for f in files:
        file_paths.append(os.path.join(directory, f))
    image_paths = []
    mask_image_paths = []
    for f in file_paths:
        mask_dir_path = os.path.join(f, "mask")
        mask_image_path = os.listdir(mask_dir_path)
        image_path = glob.glob(f+'/??????????.png')
        for m in mask_image_path:
            mask_image_paths.append(os.path.join(mask_dir_path, m))
            image_paths.append(image_path[0])

    return image_paths, mask_image_paths


def main(directory):
    image_paths, mask_image_paths = _get_dataset_paths(directory)


if __name__ == '__main__':
    args = sys.argv
    directory = args[1]
    main(directory)