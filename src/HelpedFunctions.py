import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


def read_image(path, plus12=False, make_lr=False):
    BGR_img = cv2.imread(path, cv2.IMREAD_COLOR)
    h = BGR_img.shape[0] + 12
    w = BGR_img.shape[1] + 12
    if plus12:
        BGR_img = cv2.resize(BGR_img, (w, h), interpolation=cv2.INTER_CUBIC)

    if make_lr:
        BGR_img = cv2.resize(BGR_img, (w // 3, h // 3),
                             interpolation=cv2.INTER_CUBIC)
        BGR_img = cv2.resize(BGR_img, (w, h), interpolation=cv2.INTER_CUBIC)

    float_img = BGR_img / 255
    return float_img


def modcrop(src, mod):
    sz = src.shape
    sz = sz - np.mod(sz, mod)
    img = src[0:sz[0], 0:sz[1]]
    return img


def sorted_list(path):
    tmplist = glob.glob(path)
    tmplist.sort()
    return tmplist


def save_graph(contents, xlabel, ylabel, savename, dir):
    np.save(f"{dir}/npy/{savename}", np.asarray(contents))
    plt.clf()
    plt.rcParams["font.size"] = 15
    plt.plot(contents, color="blue", linestyle="-", label="loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig(f"{dir}/graph/{savename}.png")
    plt.close()


def makedir(path):
    try:
        os.mkdir(path)
    except:
        pass


def float2uint8(img):
    result = np.squeeze(img, axis=0)
    result = result * 255
    result = result.astype("uint8")
    return result


def downgrade(image, scale=3):
    h = image.shape[0]
    w = image.shape[1]
    orig_size = (w, h)
    new_size = (w // scale, h // scale)
    new_img = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    new_img = cv2.resize(new_img, orig_size, interpolation=cv2.INTER_CUBIC)
    return new_img


