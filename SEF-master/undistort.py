import numpy as np
import cv2
import sys
from PIL import Image

# You should replace these 3 lines with the output in calibration step
DIM=(1920, 1080)
K=np.array([[1133.8017252710802, 0.0, 954.433450235042], [0.0, 1133.9433526784026, 595.6902271513281], [0.0, 0.0, 1.0]])
D=np.array([[-0.1087578718855901], [0.036920073018329364], [-0.007829401410740683], [-0.018477564004057322]])

def undistort(img_path):
    # img = cv2.imread(img_path)
    img = np.asarray(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    im = Image.fromarray(undistorted_img)
    return im
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)