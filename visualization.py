import cv2
import numpy as np

a = np.load("result.npy").reshape(-1, 68, 68)
a = np.maximum(a, 0)
a = a * 255
a = a.astype(np.uint8)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
vw = cv2.VideoWriter("result3layer.avi", fourcc, 23.18, (68,68), False)
for i in range(len(a)):
    vw.write(a[i])
vw.release()