## The trainer code. Will run the network through the training steps and save the output

import torchvision
import torchvision.transforms as transforms
import torch
import time
import torch.optim as optim
import torch.nn as nn
import datetime
import os

from losses import SqrHingeLoss
from torch.optim.lr_scheduler import MultiStepLR

class Networktrainer():
    def __init__(self, traintime,bnn_Accuracy = 85):
        # Time given to the BNN to train
        self.traintime = traintime
        print("The Amount of time that the BNN will have to train is: {} hours".format(traintime))
        self.traintime = traintime*60*60 # convert the time to seconds
        
        # The device that will be used for training
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Assuming that we are on a CUDA machine, this should print a CUDA device:

        print(f"Training on: {self.device}")

        #Print the time that the Network Trainer has started to the screen
        print(f"Training Class initalised at: {datetime.datetime.now()}")
        self.path = f"./Training_log/{datetime.datetime.now()}"
        os.mkdir(self.path)
        # Set up for the system to run without haing to run on the BNN
        self.binaryAccuracy = bnn_Accuracy


        pass
    def MirrorMNIST(self):
        return [(
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
                     (
                         "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
                         "d53e105ee54ea40749a09fcbcd1e9432"),
                     (
                         "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
                         "9fb629c4189551a2d022fa330f9573f3"),
                     (
                         "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
                         "ec29112dd5afa0611ce80d1b7f02629c")]
    def load_dataset(self,Dataset="MNIST"):
        print(f"loading dataset: {Dataset}")
        if Dataset == 'MNIST':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))])

            batch_size = 4

            trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                    download=True, transform=transform)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                    shuffle=True, num_workers=2)

            testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=True, transform=transform)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                    shuffle=False, num_workers=2)

            self.classes = ('0', '1', '2', '3',
                    '4', '5', '6', '7', '8', '9')
            

        elif Dataset == 'CIFAR10':
            transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            batch_size = 4

            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                    shuffle=True, num_workers=2)

            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                    shuffle=False, num_workers=2)

            self.classes = ('plane', 'car', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        elif Dataset == "MNIST-aug":
            batch_size = 4
            transform_to_tensor = transforms.Compose([transforms.ToTensor()])
            # an augmented version of mnist
            transform_train = transform_to_tensor
            builder = self.MirrorMNIST()
            train_set = builder(train=True, download=True, transform=transform_train)
            test_set = builder(train=False, download=True, transform=transform_to_tensor)
            self.trainloader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True, num_workers=2)
            self.testloader = torch.utils.data.DataLoader(
                test_set, batch_size=batch_size, shuffle=False, num_workers=2)
            
            self.classes = ('0', '1', '2', '3',
                    '4', '5', '6', '7', '8', '9')

        else:
            assert False, "Dataset could not be loaded"
        print("Dataset loaded")
    
    # def resumetrainNetwork(self,NetworkLocation,epochNumber:int,Binary:bool,Name:str)
    #     print(f"Resuming {Name} at {datetime.datatime.now()}. Epoch number {epochNumber}")

    # def loadNetwork(self,Name,timeStamp,Suppress_Message=True):
    #     if not Suppress_Message:
    #         print(f"Loading Network {Name} at time {timeStamp}")
        
        
    #     return Network

    def saveNetwork(self,Network,Name,timeStamp,Suppress_Message=True):
        if not Suppress_Message:
            print(f"Saving Network {Name} at time {datetime.datetime.now()}")
        Network.to("cpu")
        self.Path = f"{self.path}/{Name}.pth" # will be able to call from the parent script
        torch.save(Network.state_dict(), self.Path)
        #articular section if the code is interrupted.
    def trainNetwork(self,Network,Binary,Name,Train = True,limitTrainTime = None):  
        # TODO: Create the system to stop over fitting. - Complete

        # It is important to select the correct type of optimizer for the network. Binaries will use Adam. 
        if Binary:
            print("The network is binary")
            optimizer = optim.Adam(Network.parameters(), lr=0.02, weight_decay=0,betas=(0.9, 0.999))
            criterion = SqrHingeLoss()
            criterion = nn.CrossEntropyLoss()

        else:
            print("The network is not binary")
            optimizer = optim.SGD(Network.parameters(), lr=0.02, weight_decay=0)
            criterion = nn.CrossEntropyLoss()

        # Lower learn rate at this epoch increment
        learn_rate = 40

        # load the networks and criterion onto your device
        Network.to(self.device)
        criterion.to(self.device)

        # set network to train
        Network.train()
        criterion.train()
        milestones = [6,10,20]
        self.scheduler = MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)

        # Open up a file to log the training.
        timeStamp = datetime.datetime.now()
        fileName = f"{self.path}/{Name}.txt"
        logfile = open(fileName,'w',encoding='utf-8') # This one will be for the Epochs.

        epoch_number = 0
        max_number_epochs = 999
        print(f"Max number of epochs for the training is {max_number_epochs} for both BNN and NN.")
        top_Accuracy = self.Test_Accuracy(Network,logfile,criterion) # No training done
        self.saveNetwork(Network,Name,timeStamp)
        logfile.write(f'New Top Accuracy is: {top_Accuracy}')
      
        if Binary:
            # Start the count down for the amount of time the network has to Train.
            startTime = time.perf_counter()
            
            
            # The training cycle.
            while True and epoch_number <= max_number_epochs: # epoch number added as a fail safe
                
                timeElapsed = time.perf_counter() - startTime
                epoch_number = epoch_number + 1
                Network.to(self.device)
                # The amount of train time alotted has passed. Stop training.
                if timeElapsed >= self.traintime or not Train:
                    accuracy = self.Test_Accuracy(Network,logfile,criterion)
                    logfile.write(f"Accuracy of final network is {top_Accuracy}%\n")
                    self.binaryAccuracy = accuracy # Log this for future use
                    print(f"The Binary network had an accuracy of {self.binaryAccuracy}%.")
                    self.saveNetwork(Network,Name,timeStamp,Suppress_Message = False)
                    break

                Network.to(self.device)
                self.train_on_dataset(Network, optimizer, criterion, logfile,epoch_number)
                
                
                # The amount of train time alotted has passed. Stop training.
                if timeElapsed >= self.traintime or not Train:
                    accuracy = self.Test_Accuracy(Network,logfile,criterion)
                    logfile.write(f"Accuracy of final network is {top_Accuracy}%\n")
                    self.binaryAccuracy = accuracy # Log this for future use
                    print(f"The Binary network had an accuracy of {self.binaryAccuracy}%.")
                    # self.saveNetwork(Network,Name,timeStamp,Suppress_Message = False)
                    if current_Accuracy > top_Accuracy:
                        self.saveNetwork(Network,Name,timeStamp)
                        logfile.write(f'New Top Accuracy is: {top_Accuracy}')
                        top_Accuracy = current_Accuracy
                    break

                
                # Every epoch will check to determine if the accuracy has improved. 
                current_Accuracy = self.Test_Accuracy(Network,logfile,criterion)
                print(f"Current Accuracy is {current_Accuracy}")
                if current_Accuracy > top_Accuracy:
                    self.saveNetwork(Network,Name,timeStamp)
                    logfile.write(f'New Top Accuracy is: {top_Accuracy}')
                    top_Accuracy = current_Accuracy
                # if top_Accuracy-current_Accuracy>10: # Stop if accuracy is going 10% less then the best.
                #     print("Network Going backwards, canning training.")
                #     break

                ## FROM FINN - Decrement the learning rate
                # Set the learning rate
                if epoch_number % learn_rate == 0:
                    optimizer.param_groups[0]['lr'] *= 0.5

                
        else:
            # This will be the code to run the other networks. It will look to attempt to match the accuracy of the other network
            logfile.write(f"Started training on {datetime.datetime.now()}\n")
            startTime = time.perf_counter()
            while True: # epoch number added as a fail safe
                epoch_number = epoch_number + 1
                
                if not Train:
                    # don't train the network
                    self.saveNetwork(Network,Name,timeStamp,Suppress_Message = False)
                    break
                self.train_on_dataset(Network, optimizer, criterion, logfile,epoch_number)
                timeElapsed = time.perf_counter() - startTime
            
                # Every second epoch will check to determine if the accuracy has improved. 
                current_Accuracy = self.Test_Accuracy(Network,logfile,criterion)
                print(f"Current Accuracy is {current_Accuracy}")
                if current_Accuracy > top_Accuracy:
                    self.saveNetwork(Network,Name,timeStamp)
                    top_Accuracy = current_Accuracy
                # if top_Accuracy-current_Accuracy>10:
                #     print("Network Going backwards, canning training.")
                #     break
                if current_Accuracy >= self.binaryAccuracy:
                    logfile.write(f"Accuracy of final network is {top_Accuracy}%. After {epoch_number} epochs.\n")
                    self.saveNetwork(Network,Name,timeStamp,Suppress_Message = False)
                    break
                if not limitTrainTime == None:
                    if timeElapsed/60>limitTrainTime:
                        logfile.write(f"Accuracy of final network is {top_Accuracy}%. After {epoch_number} epochs. Reached the end of training time.\n")
                        self.saveNetwork(Network,Name,timeStamp,Suppress_Message = False)
                        break
                if epoch_number>=max_number_epochs:
                    logfile.write(f"The network excited training after reaching the max number of epochs of training. Max number of epochs: {max_number_epochs}\n")
                    logfile.write(f"Accuracy of final network is {top_Accuracy}%. After {epoch_number} epochs.\n")
                    break
                if epoch_number % learn_rate == 0:
                    optimizer.param_groups[0]['lr'] *= 0.5

            timeElapsed = time.perf_counter() - startTime
            logfile.write(f"Ended Training on {datetime.datetime.now()}. Training took {timeElapsed} seconds.")
            print(f"The Quantised network had an accuracy of {top_Accuracy}%.")
        logfile.close
        return Network

    def Test_Accuracy(self, Network, logfile,criterion=None):
        # code adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
                    # Check to see the accuracy of the system
        Network.eval()
        correct = 0
        total = 0
        Network.to(self.device)
        for data in self.testloader:
            (images, labels) =  data[0].to(self.device), data[1].to(self.device)
                # calculate outputs by running images through the network
            outputs = Network(images)
                # the class with the highest energy is what we choose as prediction
    
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        logfile.write(f"Accuracy of network is {100*correct // total}%\n")
        return 100*correct // total

    def train_on_dataset(self, Network, optimizer, criterion, logfile,epoch):
        # Go over the dataset once.
        Network.train() # just to ensure that it is set to train
        running_loss = 0.0
        Network.to(self.device)

        for i, data in enumerate(self.trainloader, 0):
            (input, target) =  data[0].to(self.device), data[1].to(self.device)

            ## Stick in the GPU code if I had a functional connection to my GPU

            output = Network(input)
            # for hingeloss only - from FINN github
            if isinstance(criterion, SqrHingeLoss):
                target = target.unsqueeze(1)
                target_onehot = torch.Tensor(target.size(0), 10).to(
                    self.device, non_blocking=True)
                target_onehot.fill_(-1)
                target_onehot.scatter_(1, target, 1)
                target = target.squeeze()
                target_var = target_onehot
            else:
                target_var = target
            loss = criterion(output, target_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            if hasattr(Network, 'clip_weights'):
                #print("Weights Clipped")
                Network.clip_weights(-1, 1)


            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                logfile.write(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 2000:.3f}\n')
                print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
