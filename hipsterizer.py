import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from imutils import face_utils
from math import atan2, degrees

## load models
detector = dlib.get_frontal_face_detector()  ## detect frontal face
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')  ## get face shape points(68 vectors)
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

## load images
# detector = dlib.cnn_face_detection_model_v1('models/bearface_network.dat')
# predictor = dlib.shape_predictor('models/landmarkDetector.dat')

img = dlib.load_rgb_image('imgs/06.jpg')
img_result = img.copy()

plt.figure(figsize=(16, 10))
plt.imshow(img)
plt.show()

## prediction
dets = detector(img)
fig, ax = plt.subplots(1, figsize=(16, 10))

for det in dets:
    x, y, w, h = det.left(), det.top(), det.width(), det.height()

    rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    shape = sp(img, det)

    for point in shape.parts():
        circle = patches.Circle((point.x, point.y), radius=3, edgecolor='r', facecolor='r')
        ax.add_patch(circle)

ax.imshow(img_result)
plt.show()


## Put glasses on
img = dlib.load_rgb_image('imgs/leemin.jpg')
img_result = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2RGBA)
glasses = cv2.imread('imgs/mask.png', cv2.IMREAD_UNCHANGED)
glasses = cv2.cvtColor(glasses, cv2.COLOR_RGB2RGBA)

dets = detector(img)

fig, ax = plt.subplots(1, figsize=(16, 10))

for det in dets:
    x, y, w, h = det.left(), det.top(), det.width(), det.height()
    
    shape = sp(img, det)
    shape_np = face_utils.shape_to_np(shape)
    
    glasses_center = np.mean([shape_np[62], shape_np[54]], axis=0).astype(int)
    glasses_size = int(np.linalg.norm(shape_np[17] - shape_np[27]) * 3.3)
    
    glasses_resized = cv2.resize(glasses.copy(), dsize=(glasses_size, glasses_size))

    def angle_between(p1, p2):
        xDiff = p2[0] - p1[0]
        yDiff = p2[1] - p1[1]
        return degrees(atan2(yDiff, xDiff))

    angle = -angle_between(shape_np[38], shape_np[44])
    M = cv2.getRotationMatrix2D((glasses_resized.shape[1] / 2, glasses_resized.shape[0] / 2), angle, 1)
    rotated_glasses = cv2.warpAffine(
        glasses_resized.copy(),
        M,
        (glasses_size, glasses_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )
    
    x_offset, y_offset = glasses_center - np.array([rotated_glasses.shape[1] / 2, rotated_glasses.shape[0] / 2], dtype=int)

    y1, y2 = y_offset, y_offset + glasses_size
    x1, x2 = x_offset, x_offset + glasses_size

    alpha_s = rotated_glasses[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        img_result[y1:y2, x1:x2, c] = (alpha_s * rotated_glasses[:, :, c] + alpha_l * img_result[y1:y2, x1:x2, c])

ax.imshow(img_result)
plt.show()