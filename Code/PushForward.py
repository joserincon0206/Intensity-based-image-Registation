import cv2
import numpy as np

def optimizeRigidTransform(fixed, moving, initAngle, initTy, initTx, writer):

    cv2.imshow('initMovin', moving)
    cv2.waitKey(5)

    sizeImage = np.shape(fixed)
    x_vector = np.arange(0, sizeImage[1])
    y_vector = np.arange(0, sizeImage[0])
    x_grid, y_grid = np.meshgrid(x_vector, y_vector)
    x_grid_slim = np.reshape(x_grid, (np.size(x_grid, )))
    y_grid_slim = np.reshape(y_grid, (np.size(y_grid, )))
    affineT = createAffine(initAngle, initTy, initTx)
    deformed_moving  = pushforward(moving, affineT)
    fixed_flat = fixed.flatten()
    deformed_moving_flat = deformed_moving.flatten()
    sqr_diff = np.power(fixed_flat - deformed_moving_flat, 2)
    SSD_error = np.sum(sqr_diff)

    count = 0
    compound_image = np.zeros((sizeImage[0], sizeImage[1], 3))
    pixels_in_moving = deformed_moving != 0
    pixels_in_fixed = fixed != 0
    pixels_in_both_fixed_moving = pixels_in_fixed *  pixels_in_moving
    alpha = 0.5
    overlaid_image = cv2.addWeighted(deformed_moving, alpha, fixed, 1-alpha, 0)

    compound_image[pixels_in_moving] = [0, 255, 255]
    compound_image[pixels_in_fixed] = [0, 255, 0]
    compound_image[pixels_in_both_fixed_moving] = [0, 0, 255]
    cv2.imshow('compound_image', overlaid_image)

    cv2.waitKey(1)


    while SSD_error > 0.0001 and count < 1000:
        affineT = createAffine(initAngle, initTx, initTy)
        deformed_moving = pushforward(moving, affineT)
        compound_image = np.zeros((sizeImage[0], sizeImage[1], 3))
        pixels_in_moving = deformed_moving != 0
        pixels_in_fixed = fixed != 0
        pixels_in_both_fixed_moving = pixels_in_fixed * pixels_in_moving
        compound_image[pixels_in_moving] = [255, 0, 0]
        compound_image[pixels_in_fixed] = [0, 255, 255]
        compound_image[pixels_in_both_fixed_moving] = [0, 255, 0]
        overlaid_image = cv2.addWeighted(deformed_moving, alpha, fixed, 1 - alpha, 0)

        ssd_string = str(SSD_error)
        ssd_string = "Registration Error: " + ssd_string
        cv2.putText(compound_image, ssd_string, (30, 30), 1, 2, (0,0,0))
        textTz = "Image Registration "
        overlaid_image = cv2.normalize(src=overlaid_image, dst=None, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX,
                                      dtype=cv2.CV_8UC1)

        overlaid_image = cv2.cvtColor(overlaid_image, cv2.COLOR_GRAY2BGR)

        cv2.putText(overlaid_image, textTz, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))


        writer.write(overlaid_image)
        cv2.imshow('compound_image', overlaid_image)
        cv2.waitKey(1)
        deformed_moving_flat = deformed_moving.flatten()
        sqr_diff = np.power(fixed_flat - deformed_moving_flat, 2)
        SSD_error = np.sum(sqr_diff)
        gradX, gradY = np.gradient(deformed_moving)
        gradX_flat = gradX.flatten()
        gradY_flat = gradY.flatten()
        sinAngle = np.sin(np.deg2rad(initAngle))
        cosAngle = np.cos(np.deg2rad(initAngle))
        jacTerm1 = gradX_flat * (-x_grid_slim * sinAngle - y_grid_slim * cosAngle) \
                     + gradY_flat * (x_grid_slim * cosAngle - y_grid_slim * sinAngle)

        jacTerm2 = gradY_flat
        jacTerm3 = gradX_flat

        jacTotal = np.zeros((np.shape(jacTerm1)[0], 3))
        jacTotal[:, 0] = jacTerm1
        jacTotal[:, 1] = jacTerm2
        jacTotal[:, 2] = jacTerm3

        error = fixed - deformed_moving

        SSD_error = np.sum(np.power(error, 2))/np.size(moving)

        error_flat = error.flatten()

        new_estimate = np.dot(np.linalg.pinv(jacTotal),error_flat)
        initAngle = initAngle + np.rad2deg(new_estimate[0])
        initTy = initTy + new_estimate[1]
        initTx = initTx + new_estimate[2]

        count += 1


    cv2.imshow('registered_image', deformed_moving)
    cv2.waitKey(0)
    return True





def pushforward(image, affineT):
    sizeImage = np.shape(image)
    x_vector = np.arange(0, sizeImage[1])
    y_vector = np.arange(0, sizeImage[0])
    x_grid, y_grid = np.meshgrid(x_vector, y_vector)
    x_grid_slim = np.reshape(x_grid, (np.size(x_grid, )))
    y_grid_slim = np.reshape(y_grid, (np.size(y_grid, )))
    xy_slim_combined = np.zeros((np.size(x_grid_slim), 3))
    xy_slim_combined[:,1] = x_grid_slim
    xy_slim_combined[:,0] = y_grid_slim
    xy_slim_combined[:, 2] = 1

    xy_slim_deformed = np.dot(affineT, xy_slim_combined.T)
    x_slim_deformed = xy_slim_deformed[1, :]
    y_slim_deformed = xy_slim_deformed[0, :]
    x_grid_deformed = np.reshape(x_slim_deformed,
                                 (np.shape(x_grid)[0],
                                  np.shape(x_grid)[1]))
    y_grid_deformed = np.reshape(y_slim_deformed,
                                 (np.shape(y_grid)[0],
                                  np.shape(y_grid)[1]))

    dstMapX, dstMapY = cv2.convertMaps(x_grid_deformed.astype(np.float32),
                                       y_grid_deformed.astype(np.float32), cv2.CV_16SC2)

    transformed_img = cv2.remap(image, dstMapY, dstMapX, cv2.INTER_LINEAR)


    return transformed_img

def createAffine(angle, Ty, Tx):
    sin_angle = np.sin(np.deg2rad(angle))
    cos_angle = np.cos(np.deg2rad(angle))
    R = np.zeros(3)
    R = np.array([[cos_angle, -sin_angle, Ty],
                  [sin_angle,  cos_angle,  Tx],
                  [ 0.      ,         0., 1.]])
    return R
