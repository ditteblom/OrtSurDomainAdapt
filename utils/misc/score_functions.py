# preamble
#from re import X
#from attr import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import torch.nn as nn

# train on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

# Define a prediction function
def predict(model, testloader):
  ''' 
  Predicts for the test data.
  '''
  predictions = []
  labels = []

  for data, target in testloader:
    data, target = data.to(device), target.to(device)
    with torch.no_grad():
      output = model(data)      

    #Append the predictions and the true labels.
    predictions.append(output.tolist())
    labels.append(target.tolist())

  #Predictions and labels are now a list of list. We want to change that.
  predictions_flat = []
  for prediction in predictions:
    for item in prediction:
      predictions_flat.append(item)

  labels_flat = []
  for label in labels:
    for item in label:
      labels_flat.append(item)

  return np.asarray(labels_flat), np.asarray(predictions_flat)

####
# The following functions is to correct for the score

def score_fluoroscopy(x): # same for all procedures
  if x < 20:
    return 0
  else:
    return -max(-1,(20-x)*(1/10))

def score_time(x):
  if x < 400:
    return 0
  else:
    return -max(-1.0,(400.0-x)*(1/100))

def score_xray(x):
  if x < 40.0:
    return 0
  else:
    return -max(-2.0,(40.0-x)*(0.1))

def score_retries_cannulated_dhs(x):
  if x < 20:
    return 0
  else:
    return max(0,2 - (x-20)*(2/5))

def score_retries_hansson(x):
  if x < 20:
    return 0
  else:
    return -max(-2,(20-x))

# For dynamic hip screw
def drill_dhs(x):
  if x < 0:
    return -max(-5,x)
  elif x < 10:
    return 0
  else:
    return -max(-5,(10-x)*(5/10))

def guidewire_dist(x):
  if x < -10:
    return -max(-7, (x+10)*3-4)
  elif x < 0:
     return -(x*(2/10)-2)
  elif x < 1:
    return -((x-1)*2)
  elif x > 3:
    return -max(-2,(3-x))
  else:
    return 0

def guidesize_cannulated(x):
  if x < 6:
    return -max(-2,(x-6)*2/2)
  elif x > 10:
    return -((10-x)*(3/10))
  else:
    return 0

def drill_dist_hansson(x):
  if x < 0:
    return -max(-13, x*8/1-5)
  elif x < 3:
    return -((x-3)*5/3)
  elif x > 5:
    return -max(-5, 5-x)
  else:
      return 0

def drill_dist_cannulated(x):
  if x < 0:
    return -max(-5-2, x*3/5-4)
  elif x < 3:
    return -((x-3)*4/3)
  elif x > 5:
    return -max(-4, (5-x)*4/5)
  else:
    return 0

def stepreamer_dist(x):
  if x < 5:
    return -max(-10, (x-5)*10/5)
  elif x > 10:
    return -max(-10, 10-x)
  else:
    return 0

class CustomInvert:
  '''
  Rotates the data if need be.
  '''
  def __init__(self, invert):
    # initializing
    self.invert = invert

  def __call__(self, x):
    #assert len(x.shape) == 3, 'x should have [nchannel x rows x cols]'
    
    if self.invert == True:
      x = transforms.functional.invert(x)
    
    return x

def saliency(input, model):
  '''
  Calculate the saliency.
  '''
  #input = input.unsqueeze(0) #Add a batch dimension
  input = input.to(device)
  input.requires_grad = True

  model.eval()

  score = model(input)
  score, indices = torch.max(score, 1) # for classification problems

  #backward pass to get gradients of score predicted class w.r.t. input image
  score.backward()

  #get max along channel axis
  slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)

  #normalize to [0..1]
  slc = (slc - slc.min())/(slc.max()-slc.min())

  #Detach
  im = input.detach().cpu().squeeze(0).numpy()
  slc = slc.cpu().numpy()

  return im, slc

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

