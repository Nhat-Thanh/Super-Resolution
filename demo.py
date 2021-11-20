from src.Model import SuperResolution
import tensorflow as tf
import os, cv2
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# nếu không có if __main__ thì kết quả chỉ được hiển thị toàn bộ sau khi train xong
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default="img.png", help='-')
    parser.add_argument('--ratio', type=int, default=2, help='-')
    FLAGS, unparsed = parser.parse_known_args()

    tf.compat.v1.disable_eager_execution()

    model = SuperResolution()
    model.restore(checkpoint_path="checkpoint/model-ckpt")

    img = cv2.imread(FLAGS.img_path)
    if img.shape != None:
        model.execute(img, FLAGS.ratio, save=True)
    else:
        print(f"{FLAGS.img_path} do not exist")
