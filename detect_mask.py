from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import pyttsx3
import pickle
import face_recognition

def give_warning(names):
	#print(names)
	speak.say( names + " please wear mask")
	speak.runAndWait()

def find_person_name( frame):
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	encodings = face_recognition.face_encodings(rgb)
	names = []
	for encoding in encodings:
		matches = face_recognition.compare_faces(data["encodings"],encoding)
		name = ""

		if True in matches:
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			
			name = max(counts, key=counts.get)
		if name not in names:
			names.append(name)
	return names


# mask detection function
def detect_and_predict_mask(frame, faceNet, maskNet):
	# find the dimensions of the frame and then construct a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []     # initialize our list of faces
	locs = []		#corresponding face locations
	preds = []		#list of predictions from our face mask network

	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the confidence is greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# to ensure the boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0: 
		# make batch predictions on all faces at the same time predictions
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	# return a 2-tuple of the face locations and their corresponding locations
	return (locs, preds)


#main function
def main():
	# load our serialized face detector model from disk
	prototxtPath = r"face_detector\deploy.prototxt"
	weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load the face mask detector model 
	maskNet = load_model("mask_detector.model")

	print("\n\nSTARTING VIDEO STREAMING...")
	vs = VideoStream(src=0).start()

	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		
		# detect faces in the frame and find if they are wearing a face mask or not
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

		name_array = []    #input frames for threads as list

		# loop over the detected face locations and their corresponding locations

		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			# determine the class label and color we'll use to draw
			# the bounding box and text
			if mask > withoutMask:
				label = "Mask on" 
			else:
				label= "No Mask"
				
			if label == "Mask on":
				colour = (0, 255, 0) 
				
			#elif label == "No Mask":
			else:
				colour = (0, 0, 255)
				name_array.append(find_person_name(frame))


			# include the perecentage in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# display the label
			cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)
			# detect the face and draw a rectangle
			cv2.rectangle(frame, (startX, startY), (endX, endY), colour, 2)
		cv2.imshow("FACE MASK DETECTION", frame)

		#print("NAME ARRAY ---",name_array)
		print(name_array)
		if name_array != None and len(name_array) > 0:
			give_warning(str(name_array))

		# press 'd' to exit
		if cv2.waitKey(1) == ord("d") or cv2.waitKey(1) == ord("D"):
			break
		

	#release resources
	cv2.destroyAllWindows()
	vs.stop()


if __name__ == "__main__":
	data = pickle.loads(open('face_enc', "rb").read())
	speak = pyttsx3.init()	
	speak.setProperty("rate", 230)
	
	main()
