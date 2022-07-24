###Test your OpenCV installation on the python emulator :

$ import cv2

_______________________________________________________________________________

###NumPy is usually imported under the np alias.

$ import cv2

###Create an alias with the as keyword while importing:

$ import numpy as np

###Now the NumPy package can be referred to as np instead of numpy.

###Matplotlib is a Python library that helps to plot graphs. It is used in data visualization and graphical plotting.

###matplotlib.pyplot is a collection of functions that make matplotlib work like MATLAB.

###Each pyplot function makes some change to a figure: e.g., creates a figure, creates a plotting area in a figure, 
###plots some lines in a plotting area, decorates the plot with labels, etc.

$ import matplotlib.pyplot as plt

________________________________________________________________________________



$ import cv2
$ net = cv2.dnn.readNetFromDarknet("## Give the address to your customized weight file")

________________________________________________________________________________

$ classes = ['Aeroplane','Balloon']

________________________________________________________________________________

###This will return video from the first webcam on your computer.

$ cap = cv2.VideoCapture(0)

$ while 1:

    _, img = cap.read()

    ###cap.read() returns a bool (True/False). 
    ###If frame is read correctly, it will be True. So you can check end of the video by checking this return value.

while 1:
    img = cv2.resize(img,(1280,720))
    ###To resize an image, OpenCV provides cv2.resize() function.

    _, img = cap.read()
    
    ###height, width, number of channels in image
    hight,width,_ = img.shape

    img = cv2.resize(img,(1280,720))


    blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)
    ###cv2. dnn. blobFromImage function returns a blob which is our input image after mean subtraction,normalizing,and channel swapping.

    hight,width,_ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)
    net.setInput(blob)


    net.setInput(blob)
    output_layers_name = net.getUnconnectedOutLayersNames()
    ###getUnconnectedOutLayers(): Get the index of the output layers.


    output_layers_name = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_name)
    ###net.forward() will give Numpy ndarray as output which you can use it to plot box on the given input image.


    layerOutputs = net.forward(output_layers_name)
    boxes =[]

    confidences = []
    boxes =[]
    class_ids = []
    confidences = []

    class_ids = []
    for output in layerOutputs:

        for detection in output:
    for output in layerOutputs:
    ####extract the class id (label) and confidence (as a probability) of
    ####the current object detection
 
            score = detection[5:]
        for detection in output:
            class_id = np.argmax(score)
            score = detection[5:]
            confidence = score[class_id]
            class_id = np.argmax(score)
            ###discard weak predictions by ensuring the detected
            ###probability is greater than the minimum probability
        
            if confidence > 0.7:
                 ###scale the bounding box coordinates back relative to the
                 ###size of the image, keeping in mind that YOLO actually
                 ###returns the center (x, y)-coordinates of the bounding
                 ###box followed by the boxes' width and height

            confidence = score[class_id]
                center_x = int(detection[0] * width)
                ###use the center (x, y)-coordinates to derive the top and
                ###and left corner of the bounding box

            if confidence > 0.7:
                center_y = int(detection[1] * hight)
                center_x = int(detection[0] * width)
                ###update our list of bounding box coordinates, confidences,
                ###and class IDs

                w = int(detection[2] * width)
                center_y = int(detection[1] * hight)
                h = int(detection[3]* hight)
                w = int(detection[2] * width)
                x = int(center_x - w/2)
                h = int(detection[3]* hight)
                y = int(center_y - h/2)
                x = int(center_x - w/2)
                boxes.append([x,y,w,h])
                y = int(center_y - h/2)
                confidences.append((float(confidence)))
                boxes.append([x,y,w,h])
                class_ids.append(class_id)
                confidences.append((float(confidence)))

                class_ids.append(class_id)


###perform the non maximum suppression given the scores defined before
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.5,.4)


    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.5,.4)
    boxes =[]

    confidences = []
    boxes =[]
    class_ids = []
    confidences = []

    class_ids = []
    for output in layerOutputs:

        for detection in output:
    for output in layerOutputs:
            score = detection[5:]
        for detection in output:
            class_id = np.argmax(score)
            score = detection[5:]
            confidence = score[class_id]
            class_id = np.argmax(score)
            if confidence > 0.5:
            confidence = score[class_id]
                center_x = int(detection[0] * width)
            if confidence > 0.5:
                center_y = int(detection[1] * hight)
                center_x = int(detection[0] * width)
                w = int(detection[2] * width)
                center_y = int(detection[1] * hight)
                h = int(detection[3]* hight)
                w = int(detection[2] * width)

                h = int(detection[3]* hight)
                x = int(center_x - w/2)

                y = int(center_y - h/2)
                x = int(center_x - w/2)

                y = int(center_y - h/2)




                boxes.append([x,y,w,h])

                confidences.append((float(confidence)))
                boxes.append([x,y,w,h])
                class_ids.append(class_id)
                confidences.append((float(confidence)))

                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.8,.4)

    font = cv2.FONT_HERSHEY_PLAIN
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.8,.4)
    colors = np.random.uniform(0,255,size =(len(boxes),3))
    font = cv2.FONT_HERSHEY_PLAIN
    if  len(indexes)>0:
    colors = np.random.uniform(0,255,size =(len(boxes),3))
        for i in indexes.flatten():

###ensure at least one detection exists

    if  len(indexes)>0:
         ###loop over the indexes we are keeping

            x,y,w,h = boxes[i]
        for i in indexes.flatten():
             ###extract the bounding box coordinates

            label = str(classes[class_ids[i]])
            x,y,w,h = boxes[i]
            confidence = str(round(confidences[i],2))
            label = str(classes[class_ids[i]])

            ###draw a bounding box rectangle and label on the image
            color = colors[i]
            confidence = str(round(confidences[i],2))
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            color = colors[i]

            ###OpenCV-Python is a library of Python bindings designed to solve computer 
            ###vision problems. cv2.putText() method is used to draw a text string on any image.
            cv2.putText(img,label + " " + confidence, (x,y+400),font,2,color,2)
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)

            cv2.putText(img,label + " " + confidence, (x,y+400),font,2,color,2)
    cv2.imshow('img',img)

    ###cv2.waitKey() returns a 32 Bit integer value
    if cv2.waitKey(1) == ord('q'):
    cv2.imshow('img',img)
        break
    if cv2.waitKey(1) == ord('q'):
    
        break

###When you call cap.release(), then:
###release software resource
###release hardware resource

cap.release()
    
###destroyAllWindows() function allows users to destroy or close all windows at any time after exiting the script.
cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()

__________________________________________________________________________________________






