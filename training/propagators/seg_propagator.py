from dense_correspondence_control.learning.training.propagators.propagator_base import PropagatorBase
import torch
from dense_correspondence_control.utils.learning_utilities import normalise_rgb


import numpy as np

class SegmentationPropagator(PropagatorBase):
    def __init__(self, loss_func, optimiser, networks):
        super().__init__(loss_func, optimiser, networks, None)

    def train_forward_backward(self, samples):
        # Get Inputs
        bottleneck_image = samples['bottleneck_rgb']
        current_image = samples['current_rgb']
        segmentation_labels = samples['current_seg']

        concatenated_image  = torch.cat([bottleneck_image,current_image],dim=1)

        # Normalise
        # concatenated_image = normalise_rgb(concatenated_image)

        # Propagate networks
        output, logits = self.networks['model'](concatenated_image)

        # Get Loss
        loss = self.loss_func(logits, segmentation_labels)

        # Clear Gradients, Backrop and Optimise
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # Return Loss
        return loss.item(), {}

    def testing_forward(self, samples):

        # Get Inputs
        bottleneck_image = samples['bottleneck_rgb']
        current_image = samples['current_rgb']
        segmentation_labels = samples['current_seg']

        concatenated_image  = torch.cat([bottleneck_image,current_image],dim=1)
        # Normalise
        # concatenated_image = normalise_rgb(concatenated_image)

        # Propagate networks
        output, logits = self.networks['model'](concatenated_image)

        # Return anything we need
        loss = self.loss_func(logits, segmentation_labels)
        mask = output > 0.5

        return loss.item(), mask  # ,  predicted_weights , individual_losses


    def real_image_testing_forward(self,bottleneck_tensor ,rgb_tensor):

        concatenated_image  = torch.cat([bottleneck_tensor,rgb_tensor],dim=1)
        # Normalise
        # concatenated_image = normalise_rgb(concatenated_image)

        # Propagate networks
        output, logits = self.networks['model'](concatenated_image)


        mask = output > 0.5

        return mask.squeeze().detach().cpu().numpy()  # ,  predicted_weights , individual_losses





class SimpleSegmentationPropagator(PropagatorBase):
    def __init__(self, loss_func, optimiser, networks, normalised):
        super().__init__(loss_func, optimiser, networks, None)
        self.normalised= normalised

    def train_forward_backward(self, samples):
        # Get Inputs
        current_image = samples['current_rgb']
        segmentation_labels = samples['current_seg']

        # Normalise
        if(self.normalised):
            current_image = normalise_rgb(current_image)

        # Propagate networks
        output, logits = self.networks['model'](current_image)

        # Get Loss
        loss = self.loss_func(logits, segmentation_labels)

        # Clear Gradients, Backrop and Optimise
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # Return Loss
        return loss.item(), {}

    def testing_forward(self, samples):

        # Get Inputs

        current_image = samples['current_rgb']
        segmentation_labels = samples['current_seg']

        # Normalise
        if(self.normalised):
            current_image = normalise_rgb(current_image)

        # Propagate networks
        output, logits = self.networks['model'](current_image)

        # Return anything we need
        loss = self.loss_func(logits, segmentation_labels)
        mask = output > 0.5

        return loss.item(), mask  # ,  predicted_weights , individual_losses


    def real_image_testing_forward(self,rgb_tensor):

        # Normalise
        if(self.normalised):
            rgb_tensor = normalise_rgb(rgb_tensor)

        # Propagate networks
        output, logits = self.networks['model'](rgb_tensor)


        mask = output > 0.5

        return mask.squeeze().detach().cpu().numpy()  # ,  predicted_weights , individual_losses

