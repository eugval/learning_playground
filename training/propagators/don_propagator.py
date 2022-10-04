from dense_correspondence_control.learning.training.propagators.propagator_base import PropagatorBase
from dense_correspondence_control.utils.learning_utilities import normalise_rgb

import torch

class DonPropagator(PropagatorBase):
    def __init__(self, loss_func, optimiser, networks):
        super().__init__(loss_func, optimiser, networks, None)


    def _forward(self, samples):
        # Get Inputs
        rgb_from = samples['rgb_from']
        rgb_to = samples['rgb_to']
        from_x = samples['from_x']
        from_y = samples['from_y']
        to_x = samples['to_x']
        to_y = samples['to_y']
        from_w_x = samples['from_w_x']
        from_w_y = samples['from_w_y']
        to_w_x = samples['to_w_x']
        to_w_y = samples['to_w_y']
        from_w_o_x = samples['from_w_o_x']
        from_w_o_y = samples['from_w_o_y']
        to_w_o_x = samples['to_w_o_x']
        to_w_o_y = samples['to_w_o_y']


        # # Normalise
        rgb_from_normalised = rgb_from #normalise_rgb(rgb_from)
        rgb_to_normalised = rgb_to #normalise_rgb(rgb_to)

        # Propagate networks
        out_from = self.networks['model'](rgb_from_normalised)
        out_to = self.networks['model'](rgb_to_normalised)

        # Get Losses
        outs_from_correct = torch.empty((out_from.shape[0], out_from.shape[1], from_x.shape[1]))
        for i in range(out_to.shape[0]):
            outs_from_correct[i] = out_from[i, :, from_y[i,:], from_x[i,:]]

        outs_to_correct = torch.empty((out_to.shape[0], out_to.shape[1], from_x.shape[1]))
        for i in range(out_to.shape[0]):
            outs_to_correct[i] = out_to[i, :, to_y[i,:], to_x[i,:]]


        outs_from_wrong = torch.empty((out_to.shape[0],  out_from.shape[1], from_w_y.shape[1]))
        for i in range(out_to.shape[0]):
            outs_from_wrong[i] = out_from[i, :, from_w_y[i,:], from_w_x[i,:]]

        outs_to_wrong = torch.empty((out_to.shape[0], out_from.shape[1], from_w_y.shape[1]))
        for i in range(out_to.shape[0]):
            outs_to_wrong[i] = out_to[i, :, to_w_y[i,:], to_w_x[i,:]]


        outs_from_obj_wrong = torch.empty((out_to.shape[0], out_from.shape[1], from_w_o_x.shape[1]))
        for i in range(out_to.shape[0]):
            outs_from_obj_wrong[i] = out_from[i, :, from_w_o_y[i,:], from_w_o_x[i,:]]

        outs_to_obj_wrong = torch.empty((out_to.shape[0], out_from.shape[1],  from_w_o_x.shape[1]))
        for i in range(out_to.shape[0]):
            outs_to_obj_wrong[i] = out_from[i, :, to_w_o_y[i,:], to_w_o_x[i,:]]

        loss, loss_correct, loss_o_wrong, loss_wrong = self.loss_func(outs_from_correct= outs_from_correct,
                                                                      outs_to_correct= outs_to_correct,
                                                                      outs_from_obj_wrong= outs_from_obj_wrong,
                                                                      outs_to_obj_wrong = outs_to_obj_wrong,
                                                                      outs_from_wrong= outs_from_wrong,
                                                                      outs_to_wrong= outs_to_wrong)

        return loss, loss_correct, loss_o_wrong, loss_wrong, out_from.detach(), out_to.detach()

    def train_forward_backward(self, samples):

        loss, loss_correct, loss_o_wrong, loss_wrong, out_from, out_to = self._forward(samples)
        # Clear Gradients, Backrop and Optimise
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        info = {
            'loss_correct': loss_correct,
            'loss_o_wrong': loss_o_wrong,
            'loss_wrong': loss_wrong,
            'out_from': out_from,
            'out_to': out_to
        }

        # Return Loss
        return loss.item() , info

    def testing_forward(self, samples):
        loss, loss_correct, loss_o_wrong, loss_wrong, out_from, out_to = self._forward(samples)

        return loss.item(), loss_correct, loss_o_wrong, loss_wrong,  out_from, out_to
