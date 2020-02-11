import cv2
import numpy as np
import os
from PIL import Image
import torch


# Labels in SIMPLE dataset range from 0 to 255
# But any pixel is supposed to be either white(255) or black(0), no gray(0 < intensity < 255)
def binarize_label(path):
    filenames = os.listdir(path)
    for filename in filenames:
        if not filename.endswith(".tiff"):
            continue
        label = np.array(Image.open(os.path.join(path, filename)))
        label[label > 122] = 255
        label[label <= 122] = 0
        label = Image.fromarray(label)

        os.remove(os.path.join(path, filename))
        label.save(os.path.join(path, filename.replace("_m", "")))


def compute_mean_and_std(path):
    filenames = os.listdir(path)
    mean, std = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    for filename in filenames:
        image = cv2.imread(os.path.join(path, filename)) / 255.0  # image shape: HxWxC
        for i in range(3):
            mean[i] += np.mean(np.reshape(image[:, :, i], -1))
            std[i] += np.std(np.reshape(image[:, :, i], -1))
    for i in range(3):
        mean[i] = mean[i] / len(filenames)
        std[i] = std[i] / len(filenames)
    print("mean:", mean)
    print("std:", std)


def crop_large_image(path, crop_size=1990):
    filenames = os.listdir(path)
    filenames.sort()
    for filename in filenames:
        raw_filename = filename.replace(".png", "")
        image = torch.from_numpy(np.array(Image.open(os.path.join(path, filename))))
        label = torch.from_numpy(np.array(Image.open(os.path.join(path.replace("train", "labels"),
                                                                  filename.replace("png", "tiff")))))
        h, w = image.shape[0], image.shape[1]
        count = 1
        starts = [0, crop_size // 2]
        for start in starts:
            for i in range(h // crop_size):
                for j in range(w // crop_size):
                    y, x = i * crop_size + start, j * crop_size + start
                    if x + crop_size > w or y + crop_size > h:
                        break
                    new_image = image[y:y+crop_size, x:x+crop_size, 0].clone()
                    new_label = label[y:y+crop_size, x:x+crop_size].clone()
                    torch.save(new_image, os.path.join(path.replace("raw_", ""), raw_filename + "_%d.pth" % count))
                    torch.save(new_label, os.path.join(path.replace("raw_", "").replace("train", "labels"),
                                                       raw_filename + "_%d.pth" % count))
                    count += 1


if __name__ == "__main__":
    '''
    binarize_label("/Users/liheyang/Downloads/U-RISC OPEN DATA SIMPLE/labels/train")
    compute_mean_and_std("/data/lihe/datasets/U-RISC/COMPLEX/raw_train")
    crop_large_image("/data/lihe/datasets/U-RISC/COMPLEX/raw_train")
    '''
    pass
