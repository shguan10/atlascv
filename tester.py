import numpy as np
import cv2

def calculate_polyfit(img):
    

def perspective_warp(img,
                     dst_size=(1242,375),
                     src=np.float32([(501,284),(701,284),(438,374),(798,374)]),
                     dst=np.float32([(438,284),(798,284),(438,374),(798,374)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

if __name__ == '__main__':
    file = 'data_road/training/gt_image_2/um_lane_000083.png'
    # image_file = 'data_road/training/image_2/um_000005.png'
    # 83, 80, 66, 55,
    img = cv2.imread(file, 1)

    dst_size = (1242,375)
    src = np.float32([(575,205),(638,205),(378,368),(785,368)])
    dst = np.float32([(575,205),(638,205),(575,368),(638,368)])

    cv2.imshow('raw_image', img)
    img = perspective_warp(img, dst_size, src, dst)

    # take out all but blue channels
    img[:,:,1] = 0
    img[:,:,2] = 0

    img = cv2.Canny(img, 100, 200)

    unwarp_img = perspective_warp(img, dst_size, dst, src)

    cv2.imshow('final_image', img)
    cv2.imshow('unwarp_img', unwarp_img)
    # cv2.imshow('unwarp', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
