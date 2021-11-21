import os, cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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

class SuperResolution:
    def __init__(self):
        self.f1 = 9
        self.c = 3
        self.n1 = 64
        self.f2 = 1
        self.n2 = 32
        self.f3 = 5

        # Định nghĩa các trọng số của mô hình,
        # chi tiết setup nằm trong paper với highlight màu đỏ
        self.W1 = tf.Variable(
            tf.random.normal(
                shape=[self.f1, self.f1, self.c, self.n1], mean=0, stddev=0.001
            ),
            name="W1",
        )

        self.B1 = tf.Variable(tf.zeros(shape=[self.n1]), name="B1")

        self.W2 = tf.Variable(
            tf.random.normal(
                shape=[self.f2, self.f2, self.n1, self.n2], mean=0, stddev=0.001
            ),
            name="W2",
        )

        self.B2 = tf.Variable(tf.zeros(shape=[self.n2]), name="B2")

        self.W3 = tf.Variable(
            tf.random.normal(
                shape=[self.f3, self.f3, self.n2, self.c],
                mean=0,
                stddev=0.001
            ),
            name="W3",
        )

        self.B3 = tf.Variable(tf.zeros(shape=[self.c]), name="B3")
        # Kết thúc định nghĩa tham số.

        self.inputs = tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=[None, None, None, None],
        )

        self.outputs = tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=[None, None, None, None],
        )

        # Định nghĩa các layer
        # patch extraction and representation layer
        self.patch_extraction = tf.nn.relu(
            tf.add(
                tf.nn.conv2d(
                    input=self.inputs,
                    filters=self.W1,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                ),
                self.B1,
            )
        )
        # non-linear mapping layer
        self.nonlinear_map = tf.nn.relu(
            tf.add(
                tf.nn.conv2d(
                    input=self.patch_extraction,
                    filters=self.W2,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                ),
                self.B2,
            )
        )

        # recontruction layer
        self.recon_tmp = tf.add(
            tf.nn.conv2d(
                input=self.nonlinear_map,
                filters=self.W3,
                strides=[1, 1, 1, 1],
                padding="SAME",
            ),
            self.B3,
        )

        self.reconstruction = tf.clip_by_value(
            self.recon_tmp, clip_value_min=0.0, clip_value_max=1.0
        )

        # Kết thúc định nghĩa layer

        # Định nghĩa hàm mất mát,
        # nó chính là MSE đã bỏ đi bước tính trung bình để loss lớn cho dễ nhìn
        self.loss = tf.sqrt(
            tf.math.reduce_sum(
                input_tensor=tf.square(self.reconstruction - self.outputs)
            )
        )
        # self.loss = tf.compat.v1.losses.mean_squared_error(
        #     labels=self.outputs, predictions=self.reconstruction
        # ) * 100

        # Định nghĩa thông số đánh giá
        """
        Tham khảo 2 link:
        (04/10/2021): https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        (04/10/2021): https://www.mathvn.com/2013/10/bang-tom-tat-cac-cong-thuc-logarit.html
        (04/10/2021): https://www.tensorflow.org/api_docs/python/tf/math/log
        (15/10/2021): https://www.mathworks.com/help/vision/ref/psnr.html
        65025 = 255^2
        """
        self.psnr = 10 * tf.experimental.numpy.log10(
            1.0 / tf.reduce_mean(
                tf.square(self.reconstruction - self.outputs)
            )
        )

        self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=1e-5
        ).minimize(self.loss)

        self.sess = tf.compat.v1.Session()
        self.saver = tf.compat.v1.train.Saver()
        self.cur_epoch = np.int32(0)
        self.train_loss_arr = np.array([])
        self.train_psnr_arr = np.array([])
        self.test_loss_arr = np.array([0])
        self.test_psnr_arr = np.array([0])

    def train(self, dataset, epochs=10, batch_size=1):
        print(f"Patch Extraction filter : {self.W1.shape}")
        print(f"Non-linear mapping      : {self.W2.shape}")
        print(f"Reconstruction          : {self.W3.shape}")

        total_epochs = self.cur_epoch + epochs
        train_list_loss = self.train_loss_arr.tolist()
        train_list_psnr = self.train_psnr_arr.tolist()
        test_list_loss = self.test_loss_arr.tolist()
        test_list_psnr = self.test_psnr_arr.tolist()
        while self.cur_epoch < total_epochs:
            isContinue = True
            while isContinue:
                x_train, y_train, isContinue = dataset.next_train_batch(
                    batch_size=batch_size
                )
                # print(f"x_train shape: {x_train.shape}")
                # print(f"y_train shape: {y_train.shape}")

                loss, psnr, _ = self.sess.run(
                    [self.loss, self.psnr, self.optimizer],
                    feed_dict={self.inputs: x_train, self.outputs: y_train},
                )

                self.saver.save(self.sess, "checkpoint/model-ckpt")

                train_list_loss.append(loss)

                train_list_psnr.append(psnr)

                # end while True
             
            self.cur_epoch += 1
            val_loss, val_psnr = self.validation(dataset)

            test_list_loss.append(val_loss)
            test_list_psnr.append(val_psnr)

            np.save("checkpoint/current_epoch", self.cur_epoch)
            print(f"Epoch [{self.cur_epoch} / {total_epochs}] | Loss: {loss}",
                  f"  PSNR: {psnr}  VAL_PSNR: {val_psnr}")

        self.train_loss_arr = np.array(train_list_loss)
        self.train_psnr_arr = np.array(train_list_psnr)
        self.test_loss_arr = np.array(test_list_loss)
        self.test_psnr_arr = np.array(test_list_psnr)
        np.save("checkpoint/train_loss_arr", self.train_loss_arr)
        np.save("checkpoint/train_psnr_arr", self.train_psnr_arr)
        np.save("checkpoint/test_loss_arr", self.test_loss_arr)
        np.save("checkpoint/test_psnr_arr", self.test_psnr_arr)
    # end train()

    def restore(self, checkpoint_path):
        # Code được tham khảo tại:
        # https://www.tensorflow.org/guide/migrate/migrating_checkpoints#setup
        if os.path.exists(f"{checkpoint_path}.index"):
            print(f'\nLOAD CHECKPOINT IN "{checkpoint_path}"\n')
            self.saver.restore(self.sess, checkpoint_path)
            self.cur_epoch = np.load("checkpoint/current_epoch.npy")
            self.train_loss_arr = np.load("checkpoint/train_loss_arr.npy")
            self.train_psnr_arr = np.load("checkpoint/train_psnr_arr.npy")
            self.test_loss_arr = np.load("checkpoint/test_loss_arr.npy")
            self.test_psnr_arr = np.load("checkpoint/test_psnr_arr.npy")
        else:
            print(f"\n{checkpoint_path} DO NOT EXIST\n")
            self.sess.run(tf.compat.v1.global_variables_initializer())
            return

        """
        Phần này tham khảo 2 link:
        (04/10/2021): https://stackoverflow.com/questions/33679382/how-do-i-get-the-current-value-of-a-variable
        (03/10/2021): https://www.tensorflow.org/guide/migrate/migrating_checkpoints#setup
        """
        reader = tf.train.load_checkpoint(checkpoint_path)
        self.sess.run(self.W1.assign(value=reader.get_tensor("W1")))
        self.sess.run(self.W2.assign(value=reader.get_tensor("W2")))
        self.sess.run(self.W3.assign(value=reader.get_tensor("W3")))
        self.sess.run(self.B1.assign(value=reader.get_tensor("B1")))
        self.sess.run(self.B2.assign(value=reader.get_tensor("B2")))
        self.sess.run(self.B3.assign(value=reader.get_tensor("B3")))

    # end restore()
    def validation(self, dataset):
        x_test, y_test = dataset.GetTestData()
        loss, psnr = self.sess.run(
            [self.loss, self.psnr],
            feed_dict={self.inputs: x_test, self.outputs: y_test},
        )
        return loss, psnr

    def test(self, dataset):
        makedir("test")
        makedir(f"test/{self.cur_epoch}")

        for test_idx in range(dataset.amount_test):
            x_test, y_test = dataset.next_test_image()
            if x_test is None:
                break

            img_recon, psnr = self.sess.run(
                [self.reconstruction, self.psnr],
                feed_dict={self.inputs: x_test, self.outputs: y_test},
            )

            img_recon = np.squeeze(img_recon, axis=0)
            img_recon = img_recon * 255
            img_recon = img_recon.astype("uint8")

            cv2.imwrite(
                "test/{}/{:05d}_psnr_({:0.2f}).png".format(
                    self.cur_epoch, test_idx, psnr
                ),
                img_recon,
            )

        X_te, y_te = dataset.GetTestData()
        loss, psnr = self.sess.run(
                [self.loss, self.psnr],
                feed_dict={self.inputs: X_te, self.outputs: y_te}
        )

        print(f"LOSS: {loss}  |  PSNR: {psnr}\n")

    # end test()

    def predict(self, img, scale=2):
        new_h = int(img.shape[0] * scale)
        new_w = int(img.shape[1] * scale)

        bicubic_img = cv2.resize(src=img, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
        img = np.expand_dims(bicubic_img, axis=0)
        float_img = img / 255
        img_recon, _ = self.sess.run(
            [self.reconstruction, self.psnr],
            feed_dict={self.inputs: float_img, self.outputs: float_img},
        )

        img_recon = np.squeeze(img_recon, axis=0)
        img_recon = img_recon * 255
        img_recon = img_recon.astype("uint8")

        return img_recon, bicubic_img

    # end execute

    # Lưu lại kết quả trong lúc train để quan sát
    # img_lr = np.expand_dims(x_train[0], axis=0)
    # img_hr = np.expand_dims(y_train[0], axis=0)

    # img_recon, tmp_psnr = self.sess.run(
    #     [self.reconstruction, self.psnr],
    #     feed_dict={self.inputs: img_lr, self.outputs: img_hr},
    # )

    # img_lr = np.squeeze(img_lr, axis=0)
    # img_recon = np.squeeze(img_recon, axis=0)
    # img_hr = np.squeeze(img_hr, axis=0)

    # plt.clf()
    # plt.rcParams["font.size"] = 100
    # plt.figure(figsize=(100, 40))
    # plt.subplot(131)
    # plt.title("Low-Resolution")
    # plt.imshow(img_lr)
    # plt.subplot(132)
    # plt.title("Reconstruction")
    # plt.imshow(img_recon)
    # plt.subplot(133)
    # plt.title("High-Resolution")
    # plt.imshow(img_hr)
    # plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    # plt.savefig(
    #     "training/{:05d}_psnr_({:0.2f}).png".format(self.cur_epoch, tmp_psnr)
    # )
    # plt.close()
    # kết thúc việc Lưu lại kết quả trong lúc train để quan sát
