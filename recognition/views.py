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

import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from django.contrib.auth.decorators import login_required
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from users.models import Attendance

from django_pandas.io import read_frame
from users.models import Attendance
import seaborn as sns
import pandas as pd
from django.db.models import Count
import mpld3
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

mpl.use('TkAgg')


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
			face_aligned = imutils.resize(face_aligned ,width = 400)
			#cv2.imshow("Image Captured",face_aligned)
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


def predict(face_aligned,svc,threshold=0.7):
	face_encodings=np.zeros((1,128))
	try:
		x_face_locations=face_recognition.face_locations(face_aligned)
		faces_encodings=face_recognition.face_encodings(face_aligned,known_face_locations=x_face_locations)
		if(len(faces_encodings)==0):
			return ([-1],[0])

	except:

		return ([-1],[0])

	prob=svc.predict_proba(faces_encodings)
	result=np.where(prob[0]==np.amax(prob[0]))
	if(prob[0][result[0]]<=threshold):
		return ([-1],prob[0][result[0]])

	return (result[0],prob[0][result[0]])


def vizualize_Data(embedded, targets,):
	X_embedded = TSNE(n_components=2).fit_transform(embedded)

	for i, t in enumerate(set(targets)):
		idx = targets == t
		plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

	plt.legend(bbox_to_anchor=(1, 1));
	
	plt.savefig('face_recognition_data/training_visualisation.png')


#saves first time_in in db
def update_attendance_in_db_in(present):
	today=datetime.date.today()
	time_in=datetime.datetime.now()
	for person in present:
		user=User.objects.get(username=person)
		try:
		   qs=Attendance.objects.get(user=user,date=today)
		except :
			qs= None
		
		if qs is None:
			if present[person]==True:
						a=Attendance(user=user,date=today,time_in=time_in,present=True,time_out=None)
						a.save()
			else:
				a=Attendance(user=user,date=today,present=False,time_in=None,time_out=None)
				a.save()
			
		else:
			if qs.present==False:
				if present[person]==True:
					#qs.date=today
					qs.time_in=time_in
					qs.present=True
					qs.save(update_fields=['date','time_in','present'])



#saves last time_out in db-testing of this function is left
def update_attendance_in_db_out(present):
	today=datetime.date.today()
	time_out=datetime.datetime.now()
	for person in present:
		user=User.objects.get(username=person)
		try:
		   qs=Attendance.objects.get(user=user,date=today)
		except :
			qs= None
		if qs is not None:
			if qs.present==True:
				if present[person]==True:
					qs.time_out=time_out
					qs.save(update_fields=['time_out'])




		
		
def no_of_emp_vs_date():
		 #today=datetime.date.today
		 #before=today-datetime.timedelta(days=7)
		 qs=Attendance.objects.filter(present='True')
		 #qs=qs.filter(date>before)
		 qs=qs.values('date').annotate(employeecount=Count('date'))
		 dataframe=read_frame(qs)
		 #fig=plt.figure()
		 sns.barplot(data=dataframe,x='date',y='employeecount')
		 plt.savefig('./recognition/static/recognition/img/graph1.png')
		 plt.close()
		 #html_graph=mpld3.fig_to_html(fig)
		 #return html_graph
		 #plt.show()

def hours_vs_date_given_employee(username):
	register_matplotlib_converters()
	user=User.objects.get(username=username)
	qs = Attendance.objects.filter(user=user)
	diff=[]
	string_date=[]

	for obj in qs:
		if obj.present==True:
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
		else:
			hours=0
		diff.append(hours)
		string_date.append(str(obj.date))

	df = read_frame(qs)
	print("df is:")
	
	df['hours']=diff
	df['strdate']=string_date
	print(df)
	
	sns.lineplot(data=df,x='strdate',y='hours')
	
	plt.savefig('./recognition/static/recognition/img/graph2.png')
	plt.close()
	

	
def hours_vs_employee_given_date(date):
	qs = Attendance.objects.filter(date=date)
	df=read_frame(qs)
	diff=[]
	username=[]

	for obj in qs:
		if obj.present==True:
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
		else:
			hours=0
		diff.append(hours)
		username.append(str(obj.user.username))


	df['hours']=diff
	df['username']=username

	sns.barplot(data=df,x='username',y='hours')
	plt.savefig('./recognition/static/recognition/img/graph3.png')
	plt.close()


	

	





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
	
	

	
	detector = dlib.get_frontal_face_detector()
	
	predictor = dlib.shape_predictor('/home/prathma/attendance_system_facial_recognition/face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	svc_save_path="face_recognition_data/svc.sav"	


		
			
	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')


	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		count[encoder.inverse_transform([i])[0]] = 0
		present[encoder.inverse_transform([i])[0]] = False

	

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
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
					
			
			(pred,prob)=predict(face_aligned,svc)
			

			
			if(pred!=[-1]):
				
				person_name=encoder.inverse_transform(np.ravel([pred]))[0]
				pred=person_name
				if count[pred] == 0:
					start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1

				if count[pred] == 4 and (time.time()-start[pred]) > 1.2:
					 count[pred] = 0
				else:
				#if count[pred] == 4 and (time.time()-start) <= 1.5:
					present[pred] = True
					log_time[pred] = datetime.datetime.now()
					count[pred] = count.get(pred,0) + 1
					print(pred, present[pred], count[pred])
				cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			else:
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			
			
			#cv2.putText()
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			#cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Mark Attendance - In - Press q to exit",frame)
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
	update_attendance_in_db_in(present)
	return redirect('home')



def mark_your_attendance_out(request):
	

	
	detector = dlib.get_frontal_face_detector()
	
	predictor = dlib.shape_predictor('/home/prathma/attendance_system_facial_recognition/face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	svc_save_path="face_recognition_data/svc.sav"	


		
			
	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')


	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		count[encoder.inverse_transform([i])[0]] = 0
		present[encoder.inverse_transform([i])[0]] = False

	

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
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
					
			
			(pred,prob)=predict(face_aligned,svc)
			

			
			if(pred!=[-1]):
				
				person_name=encoder.inverse_transform(np.ravel([pred]))[0]
				pred=person_name
				if count[pred] == 0:
					start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1

				if count[pred] == 4 and (time.time()-start[pred]) > 1.5:
					 count[pred] = 0
				else:
				#if count[pred] == 4 and (time.time()-start) <= 1.5:
					present[pred] = True
					log_time[pred] = datetime.datetime.now()
					count[pred] = count.get(pred,0) + 1
					print(pred, present[pred], count[pred])
				cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			else:
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			
			
			#cv2.putText()
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			#cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Mark Attendance- Out - Press q to exit",frame)
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
	update_attendance_in_db_out(present)
	return redirect('home')




@login_required
def train(request):
	if request.user.username!='admin':
		return redirect('not-authorised')

	training_dir='face_recognition_data/training_dataset'
	
	
	
	count=0
	for person_name in os.listdir(training_dir):
		curr_directory=os.path.join(training_dir,person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in image_files_in_folder(curr_directory):
			count+=1

	X=[]
	y=[]
	i=0


	for person_name in os.listdir(training_dir):
		print(str(person_name))
		curr_directory=os.path.join(training_dir,person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in image_files_in_folder(curr_directory):
			print(str(imagefile))
			image=cv2.imread(imagefile)
			try:
				X.append((face_recognition.face_encodings(image)[0]).tolist())
				

				
				y.append(person_name)
				i+=1
			except:
				print("removed")
				os.remove(imagefile)

			


	targets=np.array(y)
	encoder = LabelEncoder()
	encoder.fit(y)
	y=encoder.transform(y)
	X1=np.array(X)
	print("shape: "+ str(X1.shape))
	np.save('face_recognition_data/classes.npy', encoder.classes_)
	svc = SVC(kernel='linear',probability=True)
	svc.fit(X1,y)
	svc_save_path="face_recognition_data/svc.sav"
	with open(svc_save_path, 'wb') as f:
		pickle.dump(svc,f)

	
	vizualize_Data(X1,targets)
	
	messages.success(request, f'Training Complete.')

	return render(request,"recognition/train.html")


@login_required
def not_authorised(request):
	return render(request,'recognition/not_authorised.html')




def view_attendance(request):
	no_of_emp_vs_date()
	hours_vs_date_given_employee('dhvanil')
	date=datetime.date.today()
	hours_vs_employee_given_date(date)
	return render(request,"recognition/view_attendance.html")




