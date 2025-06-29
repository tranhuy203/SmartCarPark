import cv2
from ultil import detect_cars

cap = cv2.VideoCapture("data/carPark.mp4")
mask = cv2.imread("data/carParkMask.png",cv2.IMREAD_GRAYSCALE)
empty_car_park_image = cv2.imread("data/emptyCarPark.png",cv2.IMREAD_GRAYSCALE)
detect_cars(cap,mask, empty_car_park_image)