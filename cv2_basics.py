import cv2
import imutils

# Readign the image
img = cv2.imread("iron_man.jpg")

# Displaying shape 
print(img.shape)

# Displaying the datatype
print(img.dtype)

# Displaying the size
print(img.size)

# Conversion to Gray Scale
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_iron_man.jpg", grayImg)

# Displaying the Gray Scaled Image
# cv2.imshow("Iorn Man", img)
# cv2.imshow("Gray Scale Iron Man", grayImg)
# cv2.waitKey(0)

# Resizing the Image
resize = imutils.resize(img, width = 200)
cv2.imwrite("resized_iron_man.jpg", resize)

# Smoothing the Image
# Syntax : dst = cv2.GuassianBlur(src, (kernal), borderType)
blur = cv2.GaussianBlur(img, (21, 21), 0)
cv2.imwrite("Guassian_iron_man.jpg", blur)

# Thresholding the Image
# Syntax : dst = cv2.threshold(src, thresholdValue, maxValueForThreshold, binaryType)
# This returns 2 arguments, both the lines below can be used
# thresholdImg = cv2.threshold(grayImg, 220, 255, cv2.THRESH_BINARY)[1]
_, thresholdImg = cv2.threshold(grayImg, 220, 255, cv2.THRESH_BINARY)
cv2.imwrite("Threshold_iron_man.jpg", thresholdImg)

_, thresholdImg = cv2.threshold(grayImg, 150, 255, cv2.THRESH_BINARY)
cv2.imwrite("Threshold_iron_man(1).jpg", thresholdImg)

# Drawing an rectangle on a Image
# Syntax : cv2.rectangle(src, startPoint, endPoint, color, thickness)
# cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

# Showing some text on an Image
# Syntax : cv2.putText(src, text, position, font, fontSize, color, thickness)
img2 = img
cv2.putText(img2, "Iron Man", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.imwrite("Text_iron_man.jpg", img2)
# cv2.imshow("Iron Man", img2)
# cv2.waitKey(0)

# Finding Contours In An Image
# Syntax : dst = cv2.findContours(srcImgCopy, contourRetrivalMode, countourApproximationMethod)
cnts = cv2.findContours(thresholdImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(cnts)



cv2.destroyAllWindows()