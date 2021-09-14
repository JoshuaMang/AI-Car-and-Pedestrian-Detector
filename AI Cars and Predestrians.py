import cv2

# create opencv video
video = cv2.VideoCapture('India - 8698.mp4')

#create classifier
carclassifier_file = "cars.xml"
car_tracker = cv2.CascadeClassifier(carclassifier_file)

predestrianclassifier_file = 'pedestrian.xml'
predestrian_tracker = cv2.CascadeClassifier(predestrianclassifier_file)


while True:

    # read the current frame
    (read_succesful, frame) = video.read()

    # safe coding

    if read_succesful:

        #convert to gray
        grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    else:
        break


    cars = car_tracker.detectMultiScale(grayscaled_frame)
    predestrians = predestrian_tracker.detectMultiScale(grayscaled_frame)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y),(x+w, y+h), (0,0,255),2)

    for (x,y,w,h) in predestrians:
        cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,255),2)

    cv2.imshow('Car Detector',frame)

    cv2.waitKey(1)





