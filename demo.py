from src.Model import SuperResolution
import tensorflow as tf
import os, cv2
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# nếu không có if __main__ thì kết quả chỉ được hiển thị toàn bộ sau khi train xong
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=2, help='-')
    FLAGS, unparsed = parser.parse_known_args()

    tf.compat.v1.disable_eager_execution()

    model = SuperResolution()
    model.restore(checkpoint_path="checkpoint/model-ckpt")
    
    image_list = os.listdir("demo/images/")
    for imname in image_list:
        impath = os.path.join("demo/images", imname)
        image = cv2.imread(impath)

        recon_img, bicubic_img = model.predict(image, FLAGS.scale)

        cv2.imwrite(f"demo/results/recon/{imname}_recon.bmp", recon_img)
        cv2.imwrite(f"demo/results/bicubic/{imname}_bicubic.bmp", bicubic_img)
