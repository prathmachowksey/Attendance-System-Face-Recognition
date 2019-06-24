from django.shortcuts import render,redirect
from .forms import usernameForm
from django.contrib import messages
from django.contrib.auth.models import User
import cv2

#utility functions:
def username_present(username):
	if User.objects.filter(username=username).exists():
		return True
	
	return False

def create_dataset():
	cam = cv2.VideoCapture(0)
	while(True):
		ret, frame = cam.read()
		

		# Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Display the resulting frame
		cv2.imshow('frame',gray)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cam.release()
	# destroying all the windows
	cv2.destroyAllWindows()
	 

def open_camera():
	cam = cv2.VideoCapture(0)
	while(True):
		ret, frame = cam.read()
		

		# Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Display the resulting frame
		cv2.imshow('frame',gray)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cam.release()
	# destroying all the windows
	cv2.destroyAllWindows()




# Create your views here.
def home(request):
	return render(request, 'recognition/home.html')


def admin_panel(request):
	return render(request, 'recognition/admin_panel.html')

def add_photos(request):
	if request.method=='POST':
		form=usernameForm(request.POST)
		data = request.POST.copy()
		username=data.get('username')
		if username_present(username):
			create_dataset()
			return redirect('add-photos')
		else:
			messages.error(request, f'No such username found. Please register employee first.')
			return redirect('admin-panel')


	else:
		form=usernameForm()
	return render(request,'recognition/add_photos.html', {'form' : form})



def mark_your_attendance(request):
	open_camera()

	return redirect('home')

