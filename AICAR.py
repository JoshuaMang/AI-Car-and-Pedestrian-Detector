import cv2

# create opencv video
img_file = "Cars.jpg"

#create classifier
carclassifier_file = "cars.xml"

# read the image
img = cv2.imread(img_file)

#create car tracker
car_tracker = cv2.CascadeClassifier(carclassifier_file)

#convert into gray color
blackandwhite = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect cars
cars = car_tracker.detectMultiScale(blackandwhite)


#Draw rectangles around the cars

for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y),(x+w, y+h), (0,0,255),2)



cv2.imshow('AICAM', img)
cv2.waitKey()




