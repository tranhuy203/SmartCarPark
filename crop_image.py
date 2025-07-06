import cv2
input_img = cv2.imread("data/carParkMask2.png")
crop_x = 26  # điểm bắt đầu theo chiều ngang
crop_y = 11   # điểm bắt đầu theo chiều dọc
crop_width = 1286
crop_height = 780
crop_img = input_img[crop_y:crop_y+crop_height,crop_x:crop_x+crop_width]
cv2.imwrite("crop.png",crop_img)
cv2.waitKey(0)