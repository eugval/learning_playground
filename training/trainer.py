from dense_correspondence_control.utils.globals import *
from torch.utils.data import DataLoader
import time
import torch
from collections import OrderedDict
import sys

import numpy as np

from dense_correspondence_control.utils.logger import Logger

import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self,
                 network_propagator,
                 training_dataset,
                 validation_dataset,
                 mini_batch_size,

                 epochs,
                 evaluation_period,
                 model_save_period,

                 evaluation_function,
                 save_name,

                 extra_evaluation_params=None,
                 training_evaluation_function = None,

                 dataloader_workers = 0,
                 dataloader_collate_function = None

                 ):
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        self.network_propagator = network_propagator

        self.epochs = epochs
        self.evaluation_period = evaluation_period
        self.model_save_period = model_save_period

        self.mini_batch_size = mini_batch_size

        self.model_save_path = os.path.join(MODELS_PATH, save_name, 'checkpoints')
        if not (os.path.exists(self.model_save_path)):
            os.makedirs(self.model_save_path)

        self.evaluation_function = evaluation_function
        self.evaluation_function.save_name = save_name
        self.extra_evaluation_params = extra_evaluation_params if extra_evaluation_params is not None else {}

        self.training_evaluation_function = training_evaluation_function

        self.logger = None
        self.logging_keys = None
        self.save_name = save_name

        # Keep track of training
        self.start_time = 0
        self.iterations = 0
        self.checkpoint_number = 0
        self.training_losses = []
        self.validation_losses = []
        self.dataloader_workers = dataloader_workers
        self.dataloader_collate_function= dataloader_collate_function

    def get_training_data(self):

        return DataLoader(self.training_dataset, batch_size=self.mini_batch_size, shuffle=True, drop_last=True, num_workers=self.dataloader_workers, collate_fn=self.dataloader_collate_function)


    def get_validation_data(self):
        return DataLoader(self.validation_dataset, batch_size=self.mini_batch_size, drop_last=False, num_workers=self.dataloader_workers, collate_fn=self.dataloader_collate_function)

    def plot_losses(self):
        epoch_number = np.arange(len(self.training_losses)) * self.evaluation_period

        plt.figure()
        plt.title('Training and Validation losses')
        plt.xlabel('Epoch number')
        plt.ylabel('Log Loss')
        plt.grid()
        plt.plot(epoch_number, np.log(self.training_losses), '-b', label='Training loss')
        plt.plot(epoch_number, np.log(self.validation_losses), '-r', label='Validation loss')
        plt.legend()
        plt.savefig(os.path.join(MODELS_PATH, self.save_name, 'losses.png'))
        plt.close()

    def evaluate(self, training_loss=0., epoch=0):
        print('Evaluating ...')
        self.network_propagator.eval()
        train_time = time.time() - self.start_time

        training_eval_data_dict = OrderedDict({
            "training_step": self.iterations,
            "training_time": train_time,
            "epoch": epoch,
            "training_loss": training_loss,
        })

        # DO EVALUATION ###
        with torch.no_grad():
            dataloader = self.get_validation_data()
            evaluation_data_dict = self.evaluation_function(self.network_propagator, dataloader,
                                                            **self.extra_evaluation_params)

        ### LOG EVERYTHING ####
        logging_dict = OrderedDict(**training_eval_data_dict, **evaluation_data_dict)
        if (self.logger is None):
            self.logging_keys = logging_dict.keys()
            self.logger = Logger(self.logging_keys, os.path.join(MODELS_PATH, self.save_name, 'log.csv'))

        logging_data = [logging_dict[name] for name in self.logging_keys]
        self.logger.log(*logging_data)

        ### PLOT AND SAVE ####
        validation_loss = evaluation_data_dict['validation_loss']

        self.training_losses.append(training_loss)
        self.validation_losses.append(validation_loss)

        self.plot_losses()

        sys.stdout.flush()

        self.network_propagator.train()

        return validation_loss

    def train(self, lr_scheduler=None):
        print('Starting Training...')
        # Make sure networks are in training mode
        self.network_propagator.train()

        training_dataloader = self.get_training_data()
        self.iterations = 0
        self.start_time = time.time()
        for epoch in range(self.epochs):
            print('Starting Epoch {}'.format(epoch))
            self.network_propagator.set_epoch(epoch)
            training_loss = 0.
            epoch_batch_count = 0
            start = time.time()
            for samples in training_dataloader:
                print('Epoch training {:.1f}%'.format(
                    (100 * epoch_batch_count * self.mini_batch_size) / len(training_dataloader.dataset)), end='\r')
                loss, info = self.network_propagator.train_forward_backward(samples)

                if(self.training_evaluation_function is not None):
                    self.training_evaluation_function(info)


                training_loss += loss
                epoch_batch_count += 1
                self.iterations += 1

            ######################################
            print('time {}'.format(time.time() - start))
            # Evaluate
            if epoch % self.evaluation_period == 0:

                mean_training_loss = training_loss / epoch_batch_count
                validation_loss = self.evaluate(training_loss=mean_training_loss, epoch=epoch)

                if (lr_scheduler is not None):
                    lr_scheduler.step(validation_loss)

            if epoch % self.model_save_period == 0 or epoch == self.epochs - 1:
                save_dict = {
                    'epoch': epoch,
                    **self.network_propagator.get_state_dicts(),
                }
                print('Saving Checkpoint ...')
                torch.save(save_dict, os.path.join(self.model_save_path, 'checkpoint_{}.pt'.format(epoch)))
                self.checkpoint_number += 1
