# import libraries
import os
import cv2
import imutils
import time
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream
import pymongo
import requests
import datetime
client=pymongo.MongoClient('mongodb+srv://appDB:Banglore1@cluster0.opser.mongodb.net/myFirstDatabase?retryWrites=true&w=majority')
Doctor=client.Doctor_Database
Nurse=client.Nurse_Database
Patient=client.Patient_Database
Log=client.Log_Database
Record=client.Record_Database
Stock=client.Stock_Database
Stock_table=Stock.Table
Record_table=Record.Table
Log_table=Log.Table
Doctor_table=Doctor.Table
Nurse_table=Nurse.Table
Patient_table=Patient.Table
medicine_1=False
medicine_2=False
medicine_3=False
medicine_4=False
def print_text(text):
    url='http://4591-2402-3a80-13a3-6c6f-ac84-5e02-a647-bbe3.ngrok.io'
    redurl='http://586b-42-105-121-25.ngrok.io'
    global medicine_1,medicine_2,medicine_3,medicine_4
    result=Patient_table.find_one({'Name':text})
    if result==None:
        print("error")
    else:
        if result['Medicine_1']==True and medicine_1==False:
            stock1=Stock_table.find_one({'Medicine_name':'Medicine 1'})
            if stock1['Quantity']==0:
                medicine_1=True
                print("Please restock Medicine 1")
            elif stock1['Quantity']==1:
                requests.get(url+'/dispenseA')
                time.sleep(5)
                Stock_table.delete_one({'Medicine_name':'Medicine 1'})
                Stock_table.insert_one({'Medicine_name':'Medicine 1','Quantity':(stock1['Quantity']-1)})
                Log_table.insert_one({'Medication':'Medicine 1','Patient':text,'Time':datetime.datetime.now()})
                requests.get(redurl+'/turn_on_A')
            else:
                requests.get(url+'/dispenseA')
                time.sleep(5)
                Stock_table.delete_one({'Medicine_name':'Medicine 1'})
                Stock_table.insert_one({'Medicine_name':'Medicine 1','Quantity':(stock1['Quantity']-1)})
                Log_table.insert_one({'Medication':'Medicine 1','Patient':text,'Time':datetime.datetime.now()})
            print(text)
        if result['Medicine_2']==True and medicine_2==False:
            stock2=Stock_table.find_one({'Medicine_name':'Medicine 2'})
            if stock2['Quantity']==0:
                medicine_2=True
                print("Please restock Medicine 2")
            elif stock2['Quantity']==1:
                requests.get(url+'/dispenseB')
                time.sleep(5)
                Stock_table.delete_one({'Medicine_name':'Medicine 2'})
                Stock_table.insert_one({'Medicine_name':'Medicine 2','Quantity':(stock2['Quantity']-1)})
                Log_table.insert_one({'Medication':'Medicine 2','Patient':text,'Time':datetime.datetime.now()})
                requests.get(redurl+'/turn_on_B')
            else:
                requests.get(url+'/dispenseB')
                time.sleep(5)
                Stock_table.delete_one({'Medicine_name':'Medicine 2'})
                Stock_table.insert_one({'Medicine_name':'Medicine 2','Quantity':(stock2['Quantity']-1)})
                Log_table.insert_one({'Medication':'Medicine 2','Patient':text,'Time':datetime.datetime.now()})
                print(text)
        if result['Medicine_3']==True and medicine_3==False:
            stock3=Stock_table.find_one({'Medicine_name':'Medicine 2'})
            if stock3['Quantity']==0:
                medicine_3=True
                print("Please restock Medicine 3")
            elif stock3['Quantity']==1:
                requests.get(url+'/dispenseC')
                time.sleep(5)
                Stock_table.delete_one({'Medicine_name':'Medicine 3'})
                Stock_table.insert_one({'Medicine_name':'Medicine 3','Quantity':(stock3['Quantity']-1)})
                Log_table.insert_one({'Medication':'Medicine 3','Patient':text,'Time':datetime.datetime.now()})
                requests.get(redurl+'/turn_on_C')
            else:   
                requests.get(url+'/dispenseC')
                time.sleep(5)
                Stock_table.delete_one({'Medicine_name':'Medicine 3'})
                Stock_table.insert_one({'Medicine_name':'Medicine 3','Quantity':(stock3['Quantity']-1)})
                Log_table.insert_one({'Medication':'Medicine 3','Patient':text,'Time':datetime.datetime.now()})
                print(text)
        if result['Medicine_4']==True and medicine_4==False:
            stock4=Stock_table.find_one({'Medicine_name':'Medicine 4'})
            if stock4['Quantity']==0:
                medicine_4=True
                print("Please restock Medicine 4")
            elif stock4['Quantity']==1:
                requests.get(url+'/dispenseD')
                time.sleep(5)
                Stock_table.delete_one({'Medicine_name':'Medicine 4'})
                Stock_table.insert_one({'Medicine_name':'Medicine 4','Quantity':(stock4['Quantity']-1)})
                Log_table.insert_one({'Medication':'Medicine 4','Patient':text,'Time':datetime.datetime.now()}) 
                requests.get(redurl+'/turn_on_D')
            else:   
                requests.get(url+'/dispenseD')
                time.sleep(5)
                Stock_table.delete_one({'Medicine_name':'Medicine 4'})
                Stock_table.insert_one({'Medicine_name':'Medicine 4','Quantity':(stock4['Quantity']-1)})
                Log_table.insert_one({'Medication':'Medicine 4','Patient':text,'Time':datetime.datetime.now()})
                print(text)

def face_recog():

	# load serialized face detector
	print("Loading Face Detector...")
	protoPath = "face_detection_model/deploy.prototxt"
	modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	# load serialized face embedding model
	print("Loading Face Recognizer...")
	embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

	# load the actual face recognition model along with the label encoder
	recognizer = pickle.loads(open("output/recognizer", "rb").read())
	le = pickle.loads(open("output/le.pickle", "rb").read())

	# initialize the video stream, then allow the camera sensor to warm up
	print("Starting Video Stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

	# start the FPS throughput estimator
	fps = FPS().start()

	# loop over frames from the video file stream
	while True:
		# grab the frame from the threaded video stream
		frame = vs.read()

		# resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
		frame = imutils.resize(frame, width=600)
		(h, w) = frame.shape[:2]

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				# perform classification to recognize the face
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]

				# draw the bounding box of the face along with the associated probability
				text = "{}".format(name)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
				
				key = cv2.waitKey(1) & 0xFF
				if key == ord("d"):
					print_text(text)

		# update the FPS counter
		fps.update()

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# stop the timer and display FPS information
	fps.stop()
	print("Elasped time: {:.2f}".format(fps.elapsed()))
	print("Approx. FPS: {:.2f}".format(fps.fps()))

	# cleanup
	cv2.destroyAllWindows()
	vs.stop()


face_recog()