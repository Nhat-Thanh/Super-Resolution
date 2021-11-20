import os, glob
import cv2
import numpy as np
from sklearn.utils import shuffle
import src.HelpedFunctions as hf

class DataSet(object):
    def __init__(self, root_dir):
        self.DatasetPath = root_dir
        # define traing high and low resolution images lists
        ls_train_lr_imgs = os.path.join(root_dir, "train_lr", "*.png")
        self.list_train_lr = self.sorted_list(ls_train_lr_imgs)

        ls_train_hr_imgs = os.path.join(root_dir, "train_hr", "*.png")
        self.list_train_hr = self.sorted_list(ls_train_hr_imgs)
        # end define traing high and low resolution images lists

        # define testing high and low resolution images lists
        ls_test_lr_imgs = os.path.join(root_dir, "test_lr", "*.png")
        self.list_test_lr = self.sorted_list(ls_test_lr_imgs)

        ls_test_hr_imgs = os.path.join(root_dir, "test_hr", "*.png")
        self.list_test_hr = self.sorted_list(ls_test_hr_imgs)
        # end define testing high and low resolution images lists

        self.amount_train = len(self.list_train_lr)
        self.amount_test = len(self.list_test_lr)

        self.idx_train = 0
        self.idx_test = 0

        self.data_val = np.zeros((0, 1, 1, 1))
        self.labels_val = np.zeros((0, 1, 1, 1))

    # end __init__()

    def sorted_list(self, path):
        tmplist = glob.glob(path)
        tmplist.sort()

        return tmplist

    # end sorted_list()

    # TODO: turn to the next batch
    def next_train_batch(self, batch_size=1):
        data = np.zeros((0, 1, 1, 1))
        labels = np.zeros((0, 1, 1, 1))
        isContinue = True

        while data.shape[0] ^ batch_size:
            # load ảnh train và label của từng ảnh
            img_lr = hf.read_image(self.list_train_lr[self.idx_train])
            img_hr = hf.read_image(self.list_train_hr[self.idx_train])

            data_tmp = np.expand_dims(img_lr, axis=0)
            label_tmp = np.expand_dims(img_hr, axis=0)

            if data.shape[0] == 0:
                data = data_tmp
                labels = label_tmp
            elif ((data.shape[1] == data_tmp.shape[1]) and (data.shape[2] == data_tmp.shape[2]) and (data.shape[3] == data_tmp.shape[3])):
                data = np.append(data, data_tmp, axis=0)
                labels = np.append(labels, label_tmp, axis=0)

            self.idx_train += 1
            if self.idx_train >= self.amount_train:
                self.list_train_lr, self.list_train_hr = shuffle(
                    self.list_train_lr, self.list_train_hr
                )
                self.idx_train = 0
                isContinue = False
                break

        return data, labels, isContinue

    #  end next_train()

    def genarate_test(self):
        if os.path.exists(os.path.join(self.DatasetPath, "data_test.npy")):
            self.data_val = np.load(os.path.join(self.DatasetPath, "data_test.npy"))
            self.labels_val = np.load(os.path.join(self.DatasetPath, "labels_test.npy"))
            return

        print("\n==================== Generating data test ====================")
        input_size = 100 
        output_size = 100 
        c = 3
        stride = 99 

        data = np.zeros((0, input_size, input_size, c))
        labels = np.zeros((0, output_size, output_size, c))

        for imname in self.list_test_hr:
            print(imname)
            ori_img = hf.read_image(imname)
            h = ori_img.shape[0]
            w = ori_img.shape[1]
            for x in np.arange(start=0, stop=h-output_size, step=stride):
                for y in np.arange(start=0, stop=w-output_size, step=stride):
                    subim_label = ori_img[x : x + output_size, y : y + output_size]
                    labels = np.vstack([labels, [subim_label]])

        for imname in self.list_test_lr:
            print(imname)
            ori_img = hf.read_image(imname)
            h = ori_img.shape[0]
            w = ori_img.shape[1]
            for x in np.arange(start=0, stop=h-output_size, step=stride):
                for y in np.arange(start=0, stop=w-output_size, step=stride):
                    subim_in = ori_img[x : x + output_size, y : y + output_size]
                    data = np.vstack([data, [subim_in]])

        self.data_val = data
        self.labels_val = labels

        np.save(os.path.join(self.DatasetPath, "data_test.npy"), data)
        np.save(os.path.join(self.DatasetPath, "labels_test.npy"), labels)
        print("=================== End generating data test ===================")

    def next_test_image(self):
        img_lr = hf.read_image(self.list_test_lr[self.idx_test])
        img_hr = hf.read_image(self.list_test_hr[self.idx_test])

        data = np.expand_dims(img_lr, axis=0)
        label = np.expand_dims(img_hr, axis=0)

        self.idx_test += 1

        if self.idx_test >= self.amount_test:
            self.idx_test = 0
            return None, None
        return data, label
    
    def GetTestData(self):
        return self.data_val, self.labels_val


    # end next_test()
