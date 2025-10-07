import cv2 as cv


def rescale(frame,scale=0.75):
    width=int((frame.shape[1] * scale))
    height=int((frame.shape[0] * scale))
    
    dim=(width,height)
    return cv.resize(frame,dim,interpolation=cv.INTER_AREA)



img=cv.imread("Photos/test.png")
cv.imshow('Cat',img)
imgResized=rescale(img)
cv.imshow("Cat-2",imgResized)

cv.waitKey(0)