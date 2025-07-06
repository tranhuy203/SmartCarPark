import cv2
from ultil import detect_cars_using_ml
# cap = cv2.VideoCapture("data/carPark.mp4")
# mask = cv2.imread("data/carParkMask.png",cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture("data/carPark2.mp4")
mask = cv2.imread("data/carParkMask2.png",cv2.IMREAD_GRAYSCALE)
detect_cars_using_ml(cap,mask)