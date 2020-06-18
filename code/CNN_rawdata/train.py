import sys
import torch
import torch.optim
import torch.nn as nn
import logging
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from utils import *
import time


from model import *
from prepare import *
from parameters import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, validation_loader, training_batches, validation_batches = load_data('/workspace/statProject/Data/RawData_with_val/train', '/workspace/statProject/Data/RawData_with_val/validation', BATCH_SIZE, get_transforms())
model = Convolutional(NUM_CLASS, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

logging.basicConfig(level=logging.INFO,
                    filename='/workspace/statProject/CNN_rawdata/train_state.log',
                    filemode='a',
                    format='%(message)s'
                    )


def train(train_loader, num_epochs, training_batches, validation_batches, model, criterion, optimizer, batch_size, learning_rate):
    """Train network."""

    # Set model to train mode
    previous_accuracy = 0

    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        train_running_loss = 0
        train_accuracy = 0
        batch_number = 0

        if (epoch + 1) % 10 == 0:
            adjust_learning_rate(optimizer, 0.8)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if batch_number % 10 == 0:
                logging.info('Batch number {}/{}.'.format(batch_number, training_batches))

            batch_number += 1

            optimizer.zero_grad()

            # Forwards pass, then backward pass, then update weights
            predicts = model.forward(images)
            loss = criterion(predicts, labels)
            loss.backward()
            optimizer.step()

            # Get the class probabilities
            prob = torch.nn.functional.softmax(predicts, dim=1)

            # Get top probabilities
            top_prob, top_class = prob.topk(1, dim=1)

            # Comparing one element in each row of top_class with
            # each of the labels, and return True/False
            equals = top_class == labels.view(*top_class.shape)

            # Number of correct predictions
            train_accuracy += torch.sum(equals.type(torch.FloatTensor)).item()
            train_running_loss += loss.item()

        # Calculating accuracy
        # train_accuracy = (train_accuracy / train_loader.batch_sampler.sampler.num_samples * 100)
        train_accuracy = (train_accuracy / (batch_size * training_batches) * 100)
        train_running_loss = train_running_loss / training_batches
        end = time.time()

        logging.info("Epoch: {}/{}.. ".format(epoch + 1, num_epochs))
        logging.info("Training Loss: {:.3f}.. ".format(train_running_loss))
        logging.info("Training Accuracy: {:.3f}%".format(train_accuracy))
        logging.info("Training Time: {:.3f} s".format(end - start))


        validation_running_loss = 0
        validation_accuracy = 0
        # Turn off gradients for testing
        with torch.no_grad():
            # set model to evaluation mode
            model.eval()
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                probabilities = model.forward(images)
                validation_running_loss += criterion(probabilities, labels)

                # Get the class probabilities
                ps = torch.nn.functional.softmax(probabilities, dim=1)

                # Get top probabilities
                top_probability, top_class = ps.topk(1, dim=1)

                # Comparing one element in each row of top_class with
                # each of the labels, and return True/False
                equals = top_class == labels.view(*top_class.shape)

                # Number of correct predictions
                validation_accuracy += torch.sum(equals.type(torch.FloatTensor)).item()

            validation_accuracy = (validation_accuracy / (batch_size * validation_batches) * 100)
            if validation_accuracy > previous_accuracy:
                save_checkpoint(epoch, model)
                previous_accuracy = validation_accuracy
                print('Epoch {}: the validation_accuracy is {}'.format(epoch, validation_accuracy))


train(train_loader, NUM_EPOCH, training_batches, validation_batches, model, criterion, optimizer, BATCH_SIZE, LEARNING_RATE)
