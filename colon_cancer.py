### CHANGE
import numpy as np
import pdb
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import random_split
import torchvision.models as models
import random
from sklearn.metrics import accuracy_score
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections

import_model = True
dataset_num = 1000
experiment_load = 1500


def calculate_performance(predictions, labels):
    metrics_output = precision_recall_fscore_support(labels, predictions, labels=(0,1), zero_division = 0)
    return(metrics_output)

def append_metrics(metrics, value, limit):
    metrics.append(value)
    if len(metrics) > limit:
        metrics.pop(0)
    return metrics

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = 'lung_colon_image_set\\colon_image_sets'
dataset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((384,384)),
    torchvision.transforms.ToTensor()
    ])
)

torch.manual_seed(0)
random.seed(0)

dataset_ratio = dataset_num/len(dataset)
dump_size = len(dataset)-dataset_num
train_size = int((len(dataset)-dump_size)*0.8)
valid_size = (len(dataset)-dump_size)-train_size
dump_dataset,train_data,val_data = random_split(dataset,[dump_size,train_size,valid_size])


train = []
for index in train_data.indices:
    train.append(dataset.imgs[index][1])
print(collections.Counter(train))

valid = []
for index in val_data.indices:
    valid.append(dataset.imgs[index][1])
print(collections.Counter(valid))

max_val = max(collections.Counter(train).values())
weighting_list=[]
for index  in range(len(collections.Counter(train))):
    weighting_list.append(max_val/collections.Counter(train)[index])
weights = torch.FloatTensor(weighting_list).to(device)

if import_model == True:
    Type = 'transfer'
else:
    Type = 'scratch'

writer = SummaryWriter(f'tensorboard_output/colon_cancer/{Type}_colon_{dataset_num}')

class_label_mapping = dataset.class_to_idx

trainloader = torch.utils.data.DataLoader(train_data, batch_size=8,
                                          shuffle=True, num_workers=0)

validationloader = torch.utils.data.DataLoader(val_data, batch_size=9,
                                         shuffle=False, num_workers=0)


net = models.resnet18(pretrained=True)
if import_model == True:
    net.fc = torch.nn.Linear(in_features=512, out_features=3, bias=True)
    net.load_state_dict(torch.load(f'saved_models/lung_cancer/scratch_lung_{experiment_load}', map_location=torch.device(device)))
    net.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
    net.to(device)
else:
    net.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
    net.to(device)


criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(net.parameters(), lr=0.001)
itergrad = 0
training_metrics = {
    'loss': [], 'precision_aca': [], 'recall_aca': [], 'f1_aca': [], 'accuracy': [],
    'precision_benign': [], 'recall_benign': [], 'f1_benign': []
}
validation_metrics = {
    'loss': [], 'precision_aca': [], 'recall_aca': [], 'f1_aca': [], 'accuracy': [],
    'precision_benign': [], 'recall_benign': [], 'f1_benign': []
}
f = open(f'{Type}_colon_{dataset_num}.txt', 'w')

for epoch in range(15):  # loop over the dataset multiple times

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels.to(device)
        inputs.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))
        loss = criterion(outputs.to(device), labels.to(device))
        predictions = torch.argmax(outputs, dim=1).detach().cpu().numpy()

        gc.collect()

        ## calculate performance metrics
        loss.backward()
        optimizer.step()
        itergrad += 1

        precision, recall, f1, support = calculate_performance(predictions, labels.to('cpu'))
        accuracy = accuracy_score(labels.to('cpu'), predictions, normalize=True)
        
        writer.add_scalar('training loss', loss, itergrad)
        writer.add_scalar('training accuracy', accuracy, itergrad)
        for i,label in enumerate(['aca', 'benign']):
            writer.add_scalar(f'training precision {label}', precision[i], itergrad)
            writer.add_scalar(f'training recall {label}', recall[i], itergrad)
            writer.add_scalar(f'training f1 {label}', f1[i], itergrad)

            append_metrics(training_metrics['precision_'+str(label)], precision[i], 100)
            append_metrics(training_metrics['recall_'+str(label)], recall[i], 100)
            append_metrics(training_metrics['f1_'+str(label)], f1[i], 100)

        append_metrics(training_metrics['accuracy'], accuracy, 100)
        append_metrics(training_metrics['loss'], loss.item(), 100)


        if itergrad % 10 == 0:    
            valid_inputs, valid_labels = iter(validationloader).next()
            valid_inputs.to(device) 
            valid_labels.to(device) 
            valid_outputs = net(valid_inputs.to(device))
            valid_predictions = torch.argmax(valid_outputs, dim=1).detach().cpu().numpy()
            valid_loss = criterion(valid_outputs.to(device), valid_labels.to(device))

            vprecision, vrecall, vf1, vsupport = calculate_performance(valid_predictions,valid_labels.to('cpu'))
            valid_accuracy = accuracy_score(valid_labels.to('cpu'), valid_predictions, normalize=True)

            gc.collect()

            writer.add_scalar('validation loss', valid_loss, itergrad)
            writer.add_scalar('validation accuracy', valid_accuracy, itergrad)
            for i,label in enumerate(['aca', 'benign']):
                writer.add_scalar(f'validation precision {label}', vprecision[i], itergrad)
                writer.add_scalar(f'validation recall {label}', vrecall[i], itergrad)
                writer.add_scalar(f'validation f1 {label}', vf1[i], itergrad)

                append_metrics(validation_metrics['precision_'+str(label)], vprecision[i], 10)
                append_metrics(validation_metrics['recall_'+str(label)], vrecall[i], 10)
                append_metrics(validation_metrics['f1_'+str(label)], vf1[i], 10)
            
            append_metrics(validation_metrics['accuracy'], valid_accuracy, 10)
            append_metrics(validation_metrics['loss'], valid_loss.item(), 10)

            if itergrad == 10:
                lowest_vloss = valid_loss.item()

            ## when to save: 
            average_loss = sum(validation_metrics['loss'])/len(validation_metrics['loss'])
            if average_loss < lowest_vloss:
                lowest_vloss = average_loss
                path = f'saved_models/colon_cancer/{Type}_colon_{dataset_num}'
                torch.save(net.state_dict(), path)
                f.write(f'iteration: {itergrad}')
                for metric,value in validation_metrics.items():
                    average_value = sum(value)/len(value)
                    f.write(f'{metric}: {average_value} ')
                for train_metric, train_value in training_metrics.items():
                    train_average_value = sum(train_value)/len(train_value)
                    f.write(f' train_{train_metric}: {train_average_value}')
                f.write('\n')

f.close()
writer.close()
print('Finished Training')