from django.shortcuts import render,redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
# Create your views here.

@login_required
def register(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	if request.method=='POST':
		form=UserCreationForm(request.POST)
		if form.is_valid():
			form.save() ###add user to database
			messages.success(request, f'Employee registered successfully!')
			return redirect('dashboard')
		


	else:
		form=UserCreationForm()
	return render(request,'users/register.html', {'form' : form})
