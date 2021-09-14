import cv2



#create classifier
carclassifier_file = "cars.xml"

car_tracker = cv2.CascadeClassifier(carclassifier_file)


while True:

    # read the current frame
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
            raise IOError("Cannot Open Webcam")

    (read_succesful,frame) = cap.read()
    frame =cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
    cv2.imshow("Input",frame)

    # safe coding

    if read_succesful:

        #convert to gray
        grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    else:
        break


    cars = car_tracker.detectMultiScale(grayscaled_frame)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y),(x+w, y+h), (0,0,255),2)

    cv2.imshow('Car Detector',frame)

    cv2.waitKey(1)





