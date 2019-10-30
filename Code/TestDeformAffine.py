import cv2
import numpy as np
from PushForward import pushforward
from PushForward import createAffine
from PushForward import optimizeRigidTransform

imageTemplate2 = cv2.imread('cameraman.png', 0)

imageTemplate = np.zeros((480, 480))
#
# imageTemplate.astype(np.uint8)
#
# cv2.imshow('template2', imageTemplate2)
# cv2.waitKey(5)

imageTemplate = cv2.circle(imageTemplate, (320, 320), 90, 1, -1)

imageTemplate = cv2.circle(imageTemplate, (320, 280), 30, 0.5, -1)
#imageTemplate = cv2.circle(imageTemplate, (320, 360), 30, 0.5, -1)

#where_is_circle = imageTemplate !=0

#imageTemplate[where_is_circle] = imageTemplate2[where_is_circle]




cv2.imshow('template', imageTemplate)
cv2.waitKey(5)

trans_affine = createAffine(15, 10, 5)

moving_image = pushforward(imageTemplate, trans_affine)

guess_angle = -30.5

guess_translationY = -10

guess_translationX = -5

writer = cv2.VideoWriter('Calibration.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (480, 480), True)

hey = optimizeRigidTransform(imageTemplate, moving_image, guess_angle, guess_translationY, guess_translationX, writer)

cv2.destroyAllWindows()

hey  = 0