from django.shortcuts import render,redirect
from .forms import usernameForm
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import time
from attendance_system_facial_recognition.settings import BASE_DIR
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from .nn4_small_2_model import create_model
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from django.contrib.auth.decorators import login_required



#utility functions:
def username_present(username):
	if User.objects.filter(username=username).exists():
		return True
	
	return False

def create_dataset(username):
	id = username
	if(os.path.exists('face_recognition_data/training_dataset/{}/'.format(id))==False):
		os.makedirs('face_recognition_data/training_dataset/{}/'.format(id))
	directory='face_recognition_data/training_dataset/{}/'.format(id)

	# Detect face
	#Loading the HOG face detector and the shape predictpr for allignment

	print("[INFO] Loading the facial detector")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('/home/prathma/attendance_system_facial_recognition/face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	#capture images from the webcam and process and detect the face
	# Initialize the video stream
	print("[INFO] Initializing Video stream")
	vs = VideoStream(src=0).start()
	#time.sleep(2.0) ####CHECK######

	# Our identifier
	# We will put the id here and we will store the id with a face, so that later we can identify whose face it is
	
	# Our dataset naming counter
	sampleNum = 0
	# Capturing the faces one by one and detect the faces and showing it on the window
	while(True):
		# Capturing the image
		#vs.read each frame
		frame = vs.read()
		#Resize each image
		frame = imutils.resize(frame ,width = 800)
		#the returned img is a colored image but for the classifier to work we need a greyscale image
		#to convert
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#To store the faces
		#This will detect all the images in the current frame, and it will return the coordinates of the faces
		#Takes in image and some other parameter for accurate result
		faces = detector(gray_frame,0)
		#In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
		
		#Print the number of faces 
		if len(faces) > 0:
			text = "{} face(s) found".format(len(faces))
			cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
		else:
			continue


		for face in faces:
			print("inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)

			face_aligned = fa.align(frame,gray_frame,face)
			# Whenever the program captures the face, we will write that is a folder
			# Before capturing the face, we need to tell the script whose face it is
			# For that we will need an identifier, here we call it id
			# So now we captured a face, we need to write it in a file
			sampleNum = sampleNum+1
			# Saving the image dataset, but only the face part, cropping the rest
			
			if face is None:
				print("face is none")
				continue


			

			cv2.imwrite(directory+'/'+str(sampleNum)+'.jpg'	, face_aligned)
			# @params the initial point of the rectangle will be x,y and
			# @params end point will be x+width and y+height
			# @params along with color of the rectangle
			# @params thickness of the rectangle
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Add Images",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		cv2.waitKey(1)
		#To get out of the loop
		if(sampleNum>300):
			break
	
	#Stoping the videostream
	vs.stop()
	# destroying all the windows
	cv2.destroyAllWindows()

	 



def get_embedding(aligned_face,model):

	aligned_face = (aligned_face / 255.).astype(np.float32) 
	embedding=model.predict(np.expand_dims(aligned_face,axis=0))[0]
	embedding=embedding.tolist()
	return embedding

# Create your views here.
def home(request):

	return render(request, 'recognition/home.html')

@login_required
def dashboard(request):
	if(request.user.username=='admin'):
		print("admin")
		return render(request, 'recognition/admin_dashboard.html')
	else:
		print("not admin")

		return render(request,'recognition/employee_dashboard.html')

@login_required
def add_photos(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	if request.method=='POST':
		form=usernameForm(request.POST)
		data = request.POST.copy()
		username=data.get('username')
		if username_present(username):
			create_dataset(username)
			messages.success(request, f'Dataset Created')
			return redirect('add-photos')
		else:
			messages.warning(request, f'No such username found. Please register employee first.')
			return redirect('dashboard')


	else:
		

			form=usernameForm()
			return render(request,'recognition/add_photos.html', {'form' : form})



def mark_your_attendance(request):
	start = time.time()
	
	print("INFO: loading the model",format(time.time()-start,'.2f'))
	
	nn4_small2_pretrained = create_model()
	print("INFO: loading the weights",format(time.time()-start,'.2f'))
	nn4_small2_pretrained.load_weights('face_recognition_data/weights/nn4.small2.v1.h5')
	
	

	print("INFO: loading the detector",format(time.time()-start,'.2f'))
	detector = dlib.get_frontal_face_detector()
	print("INFO: loading the shape predictor",format(time.time()-start,'.2f'))
	predictor = dlib.shape_predictor('/home/prathma/attendance_system_facial_recognition/face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	svc_save_path="face_recognition_data/svc.sav"	


	print("INFO: loading the svc model",format(time.time()-start,'.2f'))	
			
	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')
	
	print("[INFO] Initializing Video stream",format(time.time()-start,'.2f'))
	vs = VideoStream(src=0).start()
	
	sampleNum = 0
	
	while(True):
		
		frame = vs.read()
		
		frame = imutils.resize(frame ,width = 800)
		
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		faces = detector(gray_frame,0)
		
		if len(faces) > 0:
			text = "{} face(s) found".format(len(faces))
			cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
		else:
			continue


		for face in faces:
			print("INFO : inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)

			face_aligned = fa.align(frame,gray_frame,face)
			
			
			embedding=get_embedding(face_aligned,nn4_small2_pretrained)
			embedding=np.array(embedding).reshape(1,-1)
			label=svc.predict(embedding)

			print("label is "+str(label))
			person_name=encoder.inverse_transform(label)[0]
			prob=svc.predict_proba(embedding)
			result=np.where(prob[0]==np.amax(prob[0]))
			if(prob[0][result[0]]<0.6):
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			else:
				cv2.putText(frame, str(person_name)+ str(prob[0][result[0]]), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
			sampleNum+=1


			

			
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			#cv2.putText()
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Add Images",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		#cv2.waitKey(1)
		#To get out of the loop
		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
	
	#Stoping the videostream
	vs.stop()
	# destroying all the windows
	cv2.destroyAllWindows()

	

	return redirect('home')




@login_required
def train(request):
	if request.user.username!='admin':
		return redirect('not-authorised')

	training_dir='face_recognition_data/training_dataset'
	X=[]
	y=[]
	nn4_small2_pretrained = create_model()
	nn4_small2_pretrained.load_weights('face_recognition_data/weights/nn4.small2.v1.h5')
	for person_name in os.listdir(training_dir):
		print(str(person_name))
		curr_directory=os.path.join(training_dir,person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in image_files_in_folder(curr_directory):
			print(str(imagefile))
			image=cv2.imread(imagefile)
			embedding=get_embedding(image,nn4_small2_pretrained)
			print(embedding)
			X.append(embedding)
			y.append(person_name)


	encoder = LabelEncoder()
	encoder.fit(y)
	y=encoder.transform(y)
	np.save('face_recognition_data/classes.npy', encoder.classes_)
	svc = SVC(probability=True)
	svc.fit(X,y)
	svc_save_path="face_recognition_data/svc.sav"
	with open(svc_save_path, 'wb') as f:
		pickle.dump(svc,f)

	cv2.destroyAllWindows()
	messages.success(request, f'Training Complete.')
	return redirect('dashboard')


@login_required
def not_authorised(request):
	return render(request,'recognition/not_authorised.html')




