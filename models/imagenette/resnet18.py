import torchvision.models as models
from torch import nn

def resnet18_nopretrained():
	#Load Resnet18 with pretrained weights
	model_ft = models.resnet18(pretrained=False)
	#Finetune Final few layers to adjust for tiny imagenet input
	model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
	num_ftrs = model_ft.fc.in_features
	model_ft.fc = nn.Linear(num_ftrs, 10)
	return model_ft

def resnet18():
	#Load Resnet18 with pretrained weights
	model_ft = models.resnet18(pretrained=True)
	#Finetune Final few layers to adjust for tiny imagenet input
	model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
	num_ftrs = model_ft.fc.in_features
	model_ft.fc = nn.Linear(num_ftrs, 10)
	return model_ft