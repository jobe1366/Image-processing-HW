import cv2 as cv
import numpy as np

### PART 1 ###
einstein = cv.imread('./einstein.jpg',cv.IMREAD_GRAYSCALE)
peppers = cv.imread('./peppers.jpg',cv.IMREAD_GRAYSCALE)

### PART 2 ###
J = np.append(peppers[:,:126],einstein[:225,126:],axis = 1)
cv.imshow('peptein',J)
cv.imwrite('peptein.jpg',J)

### PART 3 ###
J_neg = 255 - J
cv.imshow('negative',J_neg)
cv.imwrite('negative_einstein.jpg',J_neg)

### PART 4 ###
peppers_color = cv.imread('./peppers_color.png',cv.IMREAD_COLOR)
#print(np.shape(peppers_color))
blue_pepper = peppers_color[:,:,0]
green_pepper = peppers_color[:,:,1]
red_pepper = peppers_color[:,:,2]

cv.imshow('blue_pepper',blue_pepper)
cv.imwrite('blue_pepper.jpg',blue_pepper)

cv.imshow('green_pepper',green_pepper)
cv.imwrite('green_pepper.jpg',green_pepper)

cv.imshow('red_pepper',red_pepper)
cv.imwrite('red_pepper.jpg',red_pepper)
