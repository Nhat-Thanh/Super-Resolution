import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

from src.Dataset import DataSet
from src.Model import SuperResolution

# nếu không có if __main__ thì kết quả chỉ được hiển thị toàn bộ sau khi train xong
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5000, help='-')
    parser.add_argument('--batch_size', type=int, default=16, help='-')
    FLAGS, unparsed = parser.parse_known_args()

    tf.compat.v1.disable_eager_execution()

    dataset = DataSet(root_dir="dataset/")
    dataset.genarate_test()
    model = SuperResolution()
    model.restore(checkpoint_path="checkpoint/model-ckpt")
    model.train(dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch_size)
    model.test(dataset)



"""
link đã tham khảo
(03/10/2021): https://www.tensorflow.org/api_docs/python/tf/compat/v1/placeholder
(03/10/2021): https://www.tensorflow.org/api_docs/python/tf/clip_by_value
(03/10/2021): https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum
(03/10/2021): https://www.tensorflow.org/guide/migrate/migrating_checkpoints
(04/10/2021): https://github.com/udaylunawat/Deep-Learning-CNNs-in-Tensorflow-with-GPUs/blob/master/Deep%20Learning%20CNN%E2%80%99s%20in%20Tensorflow%20with%20GPUs/slides.pdf
(04/10/2021): https://www.tensorflow.org/api_docs/python/tf/image/psnr
(04/10/2021): https://www.tensorflow.org/hub/tutorials/image_enhancing
(04/10/2021): https://www.tensorflow.org/api_docs/python/tf/compat/v1/losses/mean_squared_error
(04/10/2021): https://www.tensorflow.org/guide/migrate/mirrored_strategy
(04/10/2021): https://www.tensorflow.org/api_docs/python/tf/compat/v1/InteractiveSession?hl=vi
(04/10/2021): https://arxiv.org/pdf/1512.07108.pdf%C3%A3%E2%82%AC%E2%80%9A

"""
