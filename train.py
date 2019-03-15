### Jae Kang ###
### jkang2 ###
### 11-785 Introduction to Deep Learning ### 
### Carnegie Mellon University ### 
### jkang2@andrew.cmu.edu ###
### github.com/asdf25jae ### 
### linkedin.com/in/asdf25jae ### 

import torch 
import numpy as np 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torch.nn.functional as F 
from PIL import Image
import os, sys, time, csv, datetime
from torch.utils.data import DataLoader, Dataset  
from torch.autograd import Variable 
import resnet as rs 
import hyperparams as hp  
import data_preprocessing as data_p
import mobilenetv2 as mn 


torch.backends.cudnn.enabled=False

## Center Loss parameters ## 
closs_weight = 0.1
lr_cent = 0.5
feat_dim = hp.feat_dim
similarity_threshold = hp.threshold 


## global variable for device that we're doing train time, infer time on ## 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## train model ## 
## for classification ## 

class CenterLoss(nn.Module):
	### Args:
	### num_classes (int): number of classes.
	### feat_dim (int): feature dimension.
	def __init__(self, net, num_classes, feat_dim, device=torch.device('cpu'), task="Classification"):
		super(CenterLoss, self).__init__()
		self.net = net 
		# self.train_loader = train_loader
		# self.test_loader = test_loader 
		self.task = task
		self.num_classes = num_classes ## n number of classes for final classification 
		self.feat_dim = feat_dim ## input dimensions of features (input of neural net)
		self.device = device ## device 
		self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device)) 
		## centers of the number of classes to the feature dimensions 

	def forward(self, x, labels):
		## Args:
		##  x: feature matrix with shape (batch_size, feat_dim).
		##  labels: ground truth labels with shape (batch_size).
		batch_size = x.size(0)
		distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
				torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
		distmat.addmm_(1, -2, x, self.centers.t())

		classes = torch.arange(self.num_classes).long().to(self.device)
		labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
		mask = labels.eq(classes.expand(batch_size, self.num_classes))

		dist = []
		for i in range(batch_size):
			value = distmat[i][mask[i]]
			value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
			dist.append(value)
		dist = torch.cat(dist)
		loss = dist.mean()

		return loss


#### Functions for training Center Loss #### 

def train(net, data_loader, test_loader_classify, test_loader_verify,
				optimizer_label, optimizer_closs,
				criterion_label, criterion_closs, task='Classification'):
	print("Beginning training! \n")
	net.train()
	batch_50_total = 50*hp.batch_size
	correctly_classified = 0
	total_correct = 0
	#epoch_correctly_classified = 0
	for epoch in range(hp.epochs):
		avg_loss = 0.0
		correctly_classified = 0
		epoch_start_time = time.time()
		start_time = time.time()
		
		for batch_num, (feats, labels) in enumerate(data_loader):
			feats, labels = feats.to(device), labels.to(device)

			#print("feats.shape :", feats.shape) # feature embedding from data_loader
			#print("labels.shape :", labels.shape) # label from dataloader
			optimizer_label.zero_grad()
			optimizer_closs.zero_grad()
		
			# closs_out, outputs = net(feats) 
			outputs = net(feats)
			#print("outputs :", outputs)
			#print("outputs.shape :", outputs.shape)
			## outputs are the predicted ids for face, 
			## feature are the feature embeddings
			_, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
			pred_labels = pred_labels.view(-1) # predicted labels 
			equality_tensor = torch.eq(pred_labels, labels)
			#print("equality_tensor :", equality_tensor)
			#print("equality_tensor.shape :", equality_tensor.shape)
			correctly_classified += torch.sum(equality_tensor).item()
			total_correct += torch.sum(equality_tensor).item()
			# print("correctly_classified :", correctly_classified)
			#epoch_correctly_classified += torch.sum(equality_tensor).item()
			# print("outputs :", outputs)
			# print("labels :", labels) 
			#print("closs_out.shape :", closs_out.shape)
			#print("outputs.shape :", outputs.shape)

			l_loss = criterion_label(outputs, labels.long())
			# c_loss = criterion_closs(closs_out, labels.long())
			# loss = l_loss + closs_weight * c_loss
			loss = l_loss
		
			loss.backward()
		
			optimizer_label.step()

			#print("criterion_closs :", criterion_closs)
			# print("criterion_closs.parameters() :", c)
			# by doing so, weight_cent would not impact on the learning of centers

			# optimizer_closs.step()
			#print("loss :", loss)
			#print("loss.item() :", loss.item())
		
			avg_loss += loss.item()
			#print("avg_loss :", avg_loss)

			if batch_num % 50 == 49:
				#print("batch_num :", batch_num)
				end_time = time.time()
				batch_50_runtime = end_time - start_time
				batch50_accuracy_rate = correctly_classified / batch_50_total
				print('Epoch: {}\tBatch: {}\tAccuracy Rate: {:.8f}\tAvg-Loss: {:.4f}\tTime: {:.4f}'.format(epoch+1,
									 batch_num+1, batch50_accuracy_rate, avg_loss/50, batch_50_runtime))
				avg_loss = 0.0    
				start_time = time.time()
				correctly_classified = 0
		
			torch.cuda.empty_cache()

			del feats
			del labels
			del loss

		torch.save(net.state_dict(), "models/190311_CNNmobilenet.pt")
		print("Saving model after one epoch!\n")
		epoch_end_time = time.time() 
		print("Epoch training time : {:.4f}".format(epoch_end_time - epoch_start_time))
		print("Epoch end time :", datetime.datetime.now())
		if task == 'Classification':
			val_loss, val_acc = test_classify_closs(net, test_loader_classify, criterion_label, criterion_closs)
			train_acc = total_correct / data_loader.dataset.__len__()
			print('Train Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
				format(train_acc, val_loss, val_acc))
			# print("Val Loss: {:.4f}\tVal Accuracy: {:.4f}".format(val_loss, val_acc))
			total_correct = 0
		else:
			test_verify_closs(net, test_loader_verify)


def test_classify_closs(net, test_loader, criterion_label, criterion_closs):
	# net : ResNet class
	# test_dataloader : DataLoader 
	# criterion
	net.eval()
	test_loss = []
	accuracy = 0
	total = 0

	for batch_num, (feats, labels) in enumerate(test_loader):
		#print("batch_num :", batch_num)

		feats, labels = feats.to(device), labels.to(device)
		# closs_out, outputs = net(feats)
		outputs = net(feats)

		_, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
		pred_labels = pred_labels.view(-1) # predicted labels 

		l_loss = criterion_label(outputs, labels.long()) # classification loss 
		# c_loss = criterion_closs(closs_out, feats.long()) # center loss 
		# loss = l_loss + closs_weight * c_loss
		loss = l_loss 

		accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
		total += len(labels)
		test_loss.extend([loss.item()]*feats.size()[0])
		del feats
		del labels

	mean_test_loss = np.mean(test_loss)
	accuracy_rate = accuracy / total
	print("Mean Test Classification loss :", mean_test_loss)
	print("Accuracy rate, Classification :", accuracy_rate)
	net.train()
	return mean_test_loss, accuracy_rate

def test_verify_closs(net, test_loader): 
	# dev dataloader as test_loader 
	# predict for verification task 
	net.eval() 
	test_loss = []
	accuracy = 0
	total = 0
	## use cosine similiarity as our final activation function for feature embedding output

	for batch_num, (feat1, feat2, labels) in enumerate(test_loader): 
		feat1, feat2, labels = feat1.to(device), feat2.to(device), labels.to(device)
		feature_embeddings1, _ = net(feat1)
		feature_embeddings2, _ = net(feat2)

		loss_vec = nn.CosineSimilarity(feature_embeddings1, feature_embeddings2)
		mask = loss.ge(similarity_threshold)
		accuracy += torch.sum(mask).item()
		total += len(labels)
		test_loss.extend([loss_vec.item()]*feats.size()[0])
		del feat1
		del feat2 
		del labels 

	mean_test_loss = np.mean(test_loss)
	accuracy_rate = accuracy / total 
	print("Mean Test Verification loss :", mean_test_loss) 
	print("Accuracy rate, Verification :", accuracy_rate)
	net.train()
	return mean_test_loss, accuracy_rate

def predict_classify(net, dataloader): 
	# generalize on the test set 
	# write to a classification_submission.csv file 
	f = open("classify_submission.csv", "w+")
	print("Beginning classification on test set!\n")
	net.eval()
	id = 0
	test_loss = []

	f.write("id,label\n")
	#print("length of dataloader :", dataloader.__len__())
	batch_num_total = dataloader.dataset.__len__() / hp.batch_size
	for batch_idx, (feats, __) in enumerate(dataloader):
		feats = feats.to(device)
		# no label 
		# __, out = net(feats)
		out = net(feats)
		_, pred_labels = torch.max(F.softmax(out, dim=1), 1)
		pred_labels = pred_labels.view(-1) # predicted labels 
		for instance in range(len(pred_labels)): 
			f.write(str(id) + "," + str(pred_labels[instance].item()) + "\n") # write to csv file 
			id += 1
			if batch_idx % 1000 == 0 and batch_idx != 0: 
				print("Batch num / Total number of batches : ")
				print(str(batch_idx) + " / " + str(batch_num_total)) 
		del feats 
	f.close()
	print("Done classifying on test set!\n")
	return 


def predict_verify(net, dataloader):
	## generalize on the test set 
	## for verification task, write to a verification_submission.csv file 
	f = open("verify_submission.csv", "w+")
	print("Beginning verification on test set!\n")
	id = 0
	net.eval()
	test_loss = []

	f.write("trial,score\n")
	batch_num_total = dataloader.dataset.__len__() / hp.batch_size
	for batch_idx, (img1, img2) in enumerate(dataloader): 
		img1, img2 = img1.to(device), img2.to(device)
		# no label 
		_ = net(img1)
		feat_embedding1 = net.feature_embedding
		_ = net(img2)
		feat_embedding2 = net.feature_embedding
		print("feat_embedding1 :", feat_embedding1)
		print("feat_embedding2 :", feat_embedding2)
		similarity_score = nn.CosineSimilarity(feat_embedding1, feat_embedding2)
		print("similarity_score :", similarity_score)
		for instance in range(hp.batch_size):
			img1_name = (dataloader.dataset.data_list[id][0])[30:] # extract name without preceding path name for img1 
			img2_name = (dataloader.dataset.data_list[id][1])[30:] # same for img2
			score = similarity_score[instance].item()
			f.write(img1_name + " " + img2_name + "," + str(score) + "\n")
			id += 1
			if batch_idx % 1000 == 0 and batch_idx != 0: 
				print("Batch num / Total number of batches : ")
				print(str(batch_idx) + " / " + str(batch_num_total)) 			 
		del img1, img2
	f.close()
	print("Done with verification on test set!\n")
	return None 

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight.data)

def main(): 

	# clear cache from past iterations
	torch.cuda.empty_cache()

	local_mode = False 
	load_pretrained = True

	parsed_dev_verify_txt = data_p.parse_txt_verify("hw2p2_check/validation_trials_verification.txt")
	parsed_test_verify_txt = data_p.parse_txt_verify("hw2p2_check/test_trials_verification_student.txt", test_mode=True)
	# above np.arrays of each image pair and label 


	transformation = torchvision.transforms.Compose(
					[torchvision.transforms.RandomVerticalFlip(),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=[0.4388, 0.2764, 0.3725],std=[0.2260, 0.1539, 0.2018])])

	# import necessary data 
	train_dataset = torchvision.datasets.ImageFolder(root='hw2p2_check/train_data/medium/', 
													transform=transformation)

	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hp.batch_size, 
											shuffle=True, num_workers=hp.num_workers)

	print("train_dataset.__len__() :", train_dataset.__len__())

	dev_dataset_classify = torchvision.datasets.ImageFolder(root='hw2p2_check/validation_classification/medium/', 
											transform=transformation)

	dev_dataloader_classify = torch.utils.data.DataLoader(dev_dataset_classify, batch_size=hp.batch_size, 
											shuffle=True, num_workers=hp.num_workers)

	dev_dataset_verify = data_p.ImageVerificationDataset(parsed_dev_verify_txt)

	dev_dataloader_verify = torch.utils.data.DataLoader(dev_dataset_verify, batch_size=hp.batch_size,
											shuffle=True, num_workers=hp.num_workers)

	test_dataset_classify = torchvision.datasets.ImageFolder(root='hw2p2_check/test_classification',
											transform=transformation)

	test_dataloader_classify = torch.utils.data.DataLoader(test_dataset_classify, batch_size=hp.batch_size,
											shuffle=False, num_workers=hp.num_workers)
	
	test_dataset_verify = data_p.ImageVerificationDataset(parsed_test_verify_txt, test_mode=True)

	test_dataloader_verify = torch.utils.data.DataLoader(test_dataset_verify, batch_size=hp.batch_size, 
											shuffle=False, num_workers=hp.num_workers)

	# net = rs.ResNet50() 
	net = mn.MobileNetV2()
	#net.apply(init_weights)

	if load_pretrained:
		net.load_state_dict(torch.load("models/190311_CNNmobilenet.pt"))
		net.eval()

	## import ResNet50 method that returns us our predefined ResNet34 model, with 
	## xavier weight init already applied 

	criterion_label = nn.CrossEntropyLoss()
	optimizer_label = torch.optim.Adam(net.parameters(), lr=hp.learning_rate)

	criterion_closs = CenterLoss(net, hp.num_classes, hp.feat_dim, device)
	optimizer_closs = torch.optim.Adam(criterion_closs.parameters(), lr=lr_cent)

	#net.train()
	net.to(device)



	## train net with joint loss function of softmax cross entropy and center loss 
	# to train classifier and feature embedding producing net simultaneously
	#train(net, train_dataloader, dev_dataloader_classify, dev_dataloader_verify,
	#			optimizer_label, optimizer_closs, criterion_label, criterion_closs)

	# test the performance of classifier and feature embedder on validation set 


	
	#test_classify(net, dev_dataloader_classify, criterion_label, criterion_closs)

	# test_verify(net, dev_dataloader_verify, criterion, )
	
	# generalize on test set 

	predict_classify(net, test_dataloader_classify)

	predict_verify(net, test_dataloader_verify)


	return None 

if __name__ == "__main__": 
	main()