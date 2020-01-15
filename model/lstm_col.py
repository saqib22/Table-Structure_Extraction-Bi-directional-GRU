import torch
import torch.nn as nn
import numpy as np;
import random;
import os;
import cv2 as cv;
import pickle

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

files = [];

for file in os.listdir("./train_col_dicts"):
	files.append(file)

class DataLoader:
	def __init__(self, path, gt_path):
		
		self.currentBatch = 0;
		self.no_of_files = len(files);
		self.path = path;
		self.gt_path = gt_path
		self.reset()

	def loadNext(self):
		# lower_bound = (self.size*self.currentBatch);
		# upper_bound = ((self.size * (self.currentBatch+1)));
		end = False;
		global files;
				
		image = files[self.currentBatch];
		
		img_dict = np.load("./train_col_dicts/"+image).item()
		input_cols=(img_dict.get("cols").astype("uint8")).squeeze()
		gt=(img_dict.get("gt"))
	
		gt_one_hot = np.zeros((1600, 2));
		gt_one_hot[np.arange(1600), gt.astype(np.uint8)] = 1;

		self.currentBatch += 1;
		if (self.currentBatch>= self.no_of_files):
		    end = True;

		return (torch.from_numpy((input_cols)).view(1,1600, 512).permute(1,0,2).contiguous().double().cuda(), torch.from_numpy((gt)).long().cuda(), end);

	def reset(self):
		print("resetting data loader");
		global files
		self.currentBatch = 0;
		random.shuffle(files)

class LSTM(nn.Module):

	def __init__(self, input_dim, hidden_dim, batch_size, output_dim=2,
					num_layers=2):
		super(LSTM, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.num_layers = num_layers

		#Define CNN layer

		# Define the LSTM layer
		self.lstm = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, bidirectional = True, dropout = 0.2)
		self.fc = nn.Linear(hidden_dim * 2, 2)
		self.softmax = nn.LogSoftmax(dim=1)

		# Define the output layer
		#self.linear = nn.Linear(self.hidden_dim, output_dim)

	def init_hidden(self):
		# This is what we'll initialise our hidden state as
		return (torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).double().cuda())
				

	def forward(self, input):
		# Forward pass through LSTM layer
		# shape of lstm_out: [input_size, batch_size, hidden_dim]
		# shape of self.hidden: (a, b), where a and b both 
		# have shape (num_layers, batch_size, hidden_dim).

		out, self.hidden = self.lstm(input, self.hidden)
		print("Shape after GRU:", out.shape)
		out = out.permute(1,0,2).contiguous().view(-1,512*2);
		print(out.shape);
		out = self.fc(out)
		print("Shape after FC:", out.shape)
		out = self.softmax(out)
		print("Shape after Softmax:", out.shape)	   
		
		# Only take the output from the final timetep
		# Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
		#y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
		#return y_pred.view(-1)
		
		return out

lstm_input_size = 512
h1 = 512
num_train = 1
output_dim = 2
num_layers = 2

print("creating model")
model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim= output_dim, num_layers=num_layers)
model.double()
model.cuda();
print("model created")

loss_fn = torch.nn.NLLLoss(weight = torch.tensor([0.66, 1.0]).double().cuda());
#loss_fn = torch.nn.BCELoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.0005)

#####################
# Train model
#####################
num_epochs = 5000;

hist = np.zeros(num_epochs)
acc = np.zeros(num_epochs)

dataLoader = DataLoader("./images_resized", "./gt_resized")

for t in range(num_epochs):
	print("epoch #:", t);

	# TO DO: Reset dataloader;
	dataLoader.reset();

	# TO DO: ITERATE OVER BATCHES
	c = 0;
	while True:

		c+=1;
		# Clear stored gradient
		#model.zero_grad()
		optimiser.zero_grad()
		
		# Initialise hidden state
		# Don't do this if you want your LSTM to be stateful
		model.hidden = model.init_hidden()
		
		X_train, y_train, end = dataLoader.loadNext();
		
		# Forward pass
		X_train.double();
		X_train.cuda();
		y_train.cuda();

		print("X_train shape", X_train.shape);
		print("Y_train shape", y_train.shape)

		y_pred = model.forward(X_train);
		y_pred.cuda()

		print("y_pred shape", y_pred.shape)

		loss = loss_fn(y_pred, y_train)

		# Zero out gradient, else they will accumulate between epochs

		# Backward pass
		loss.backward()

		# Update parameters
		optimiser.step()
		if (end):
			break;
	if t % 5 == 0:
		print("Epoch ", t, "ERROR: ", loss.item())
	hist[t] = loss.item()
	np.save("./loss.npy", hist)

	if t % 5 == 0:
		torch.save(model.state_dict(), "./model"+str(t)+".pth");

		total_correct = 0;
		count = [0,0]
		for file in os.listdir("./val_col_dicts"):
		    img_dict = np.load("./val_col_dicts/" + file).item()
		    input_cols=(img_dict.get("cols").astype("uint8")).squeeze()
		    input_cols = torch.from_numpy((input_cols)).view(1,1600, 512).permute(1,0,2).contiguous().double().cuda()
		    
		    with torch.no_grad():
		        y_pred = model.forward(input_cols);
		        y_pred = torch.argmax(y_pred, dim=1)
		        gt = (img_dict.get("gt"))
		        
		        y_arr = y_pred.cpu().numpy()
		        correct = 0;

		        for i in range(1600):
		            if (gt[i] == y_arr[i]):
		                correct +=1
		            count[y_arr[i]] += 1;
		        total_correct += correct;
		print("Accuracy:", total_correct/(1600*14))
		acc[t] = total_correct/(1600*14);
		np.save("./accuracy.npy", acc)