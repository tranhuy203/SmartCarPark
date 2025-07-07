import cv2
import numpy as np
import cv2
import numpy as np

def extract_roi_by_rotated_rect(image, contour, edge_map):
    """
    Trích xuất vùng ảnh của contour theo hình chữ nhật xoay khít nhất.

    Tham số:
        image   : ảnh gốc (ndarray)
        contour : một contour (ndarray)

    Trả về:
        roi     : ảnh vùng contour đã xoay và cắt khít (ndarray)
    """
    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    size = tuple([int(s) for s in size])  # Làm tròn kích thước

    # Ma trận xoay
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Xoay toàn bộ ảnh
    rotated = cv2.warpAffine(image, M, image.shape[1::-1], flags=cv2.INTER_CUBIC)
    edge_rotated = cv2.warpAffine(edge_map, M, image.shape[1::-1], flags=cv2.INTER_CUBIC)

    # Cắt vùng ảnh đã xoay
    x = int(center[0] - size[0] / 2)
    y = int(center[1] - size[1] / 2)
    roi = rotated[y:y + size[1], x:x + size[0]]
    map = edge_rotated[y:y + size[1], x:x + size[0]]
    return roi, map

img = cv2.imread('data/carParkMask.png',0)
bg = cv2.imread("frame.png")
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
edge_map = cv2.Canny(bg,50,150)
roi, edge = extract_roi_by_rotated_rect(bg,contours[1],edge_map)
cv2.imshow("fadsfd",roi)
cv2.imshow("fadfsfd",edge)
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(bg, (x, y), (x + w, y + h), (0, 0, 0), 1)  # Màu xanh lá

    # cv2.drawContours(bg,[contour],-1,(0,0,255),1)

    # rect = cv2.minAreaRect(contour)
    # print(rect[2])
    # box = cv2.boxPoints(rect)
    # box = box.astype(np.intp)
    # cv2.drawContours(bg, [box], 0, (0, 0, 0), 1)  # Màu đỏ

# cv2.imshow("img",bg)
# cv2.imshow("mask",img)
# for i, cnt in enumerate(contours):
#     rect = cv2.minAreaRect(cnt)
#     box = cv2.boxPoints(rect)
#     box = np.int32(box)
#
#     # Tính ma trận xoay
#     center, size, angle = rect
#     size = tuple([int(s) for s in size])
#
#     # Tạo ma trận biến đổi xoay
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#
#     # Xoay toàn bộ ảnh
#     rotated = cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_CUBIC)
#
#     # Cắt vùng chữ nhật khít sau khi xoay
#     x, y = int(center[0] - size[0] / 2), int(center[1] - size[1] / 2)
#     roi = rotated[y:y + size[1], x:x + size[0]]
#
#     # Hiển thị hoặc lưu vùng ROI
#     cv2.imshow(f'Contour {i}', roi)

cv2.waitKey(0)
