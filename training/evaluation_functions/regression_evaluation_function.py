import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from dense_correspondence_control.utils.globals import MODELS_PATH
from dense_correspondence_control.utils.learning_utilities import min_max_stardardisation_to_01
import cv2



def test_direct_reg(network_propagator, dataloader, **kwargs):

    device = kwargs['device']
    use_sin_cos = kwargs['use_sin_cos']
    res = {}

    ### Sim evaluation
    counter_val_loss = 0
    validation_batch_counter = 0.
    validation_loss = 0.
    sim_pixel_error = 0.
    sim_pixel_error_std = 0.
    sim_angle_error = 0.
    sim_angle_error_std = 0.
    sim_scale_error = 0.
    sim_scale_error_std = 0.



    # Scatter plot accumulations
    angles_error_scatter = []
    angles_label_scatter = []

    scales_error_scatter = []
    scales_label_scatter = []

    pixel_errors_scatter = []
    pixel_labels_scatter = []


    normaliser = network_propagator.normaliser
    out_types = network_propagator.output

    for i, samples in enumerate(dataloader):
        loss, outputs = network_propagator.testing_forward(samples)
        # accumulate loss
        validation_loss += loss

        scales_index = 0
        pixel_deltas_index = 0
        #TODO: indexing here assumes that the [z_angels, scales, pixel_deltas] always appear in that order, even if some are missing
        if('z_angles' in out_types):
            if(use_sin_cos):
                scales_index += 2
                pixel_deltas_index += 2
                # z angle error
                z_angle_predictions = outputs.detach()[:,:2]
                z_angle_predictions = normaliser.decode(z_angle_predictions)

                z_angle_labels = samples['z_angles']
                z_angle_labels = normaliser.decode(z_angle_labels)
            else:
                scales_index += 1
                pixel_deltas_index += 1
                # z angle error
                z_angle_predictions = outputs.detach()[:, 0]
                z_angle_predictions = normaliser.un_normalise(z_angle_predictions,'z_angles')

                z_angle_labels = samples['z_angles'][:, 0]
                z_angle_labels = normaliser.un_normalise(z_angle_labels,'z_angles')

            z_angle_error_unormalised = torch.abs(z_angle_predictions - z_angle_labels)
            z_angle_error = z_angle_error_unormalised / torch.abs(z_angle_labels)
            z_angle_error_mean = torch.mean(z_angle_error)
            z_anlge_error_std =  torch.std(z_angle_error)

            sim_angle_error += z_angle_error_mean.item()
            sim_angle_error_std += z_anlge_error_std.item()

            angles_error_scatter +=  [a.item()*180./np.pi for a in z_angle_error_unormalised]
            angles_label_scatter += [a.item()*180./np.pi for a in z_angle_labels]

        if('scales' in out_types):
            pixel_deltas_index += 1
            # scale error
            scale_predictions = outputs.detach()[:,scales_index].unsqueeze(1)
            scale_predictions = normaliser.un_normalise(scale_predictions, 'scales')

            scales_labels = samples['scales']
            scales_labels = normaliser.un_normalise(scales_labels, 'scales')

            scale_error_unormalised = torch.abs(scale_predictions-scales_labels)
            scale_error = scale_error_unormalised / torch.abs(scales_labels)
            scale_error_mean = torch.mean(scale_error)
            scale_error_std = torch.std(scale_error)

            sim_scale_error+=scale_error_mean.item()
            sim_scale_error_std+=scale_error_std.item()

            scales_error_scatter += [s.item() for s in scale_error_unormalised]
            scales_label_scatter += [s.item() for s in scales_labels]

        if('pixel_deltas' in out_types):
            # pixel error
            pixel_delta_predictions = outputs.detach()[:,pixel_deltas_index:]
            pixel_delta_predictions = normaliser.un_normalise(pixel_delta_predictions, 'pixel_deltas')

            pixel_deltas_labels = samples['pixel_deltas']
            pixel_deltas_labels = normaliser.un_normalise(pixel_deltas_labels, 'pixel_deltas')

            pixel_error_unormalised = torch.norm(pixel_delta_predictions-pixel_deltas_labels, dim = 1)
            pixel_error =  pixel_error_unormalised/  torch.norm(pixel_deltas_labels, dim = 1)
            pixel_error_mean = torch.mean(pixel_error)
            pixel_error_std = torch.std(pixel_error)

            sim_pixel_error += pixel_error_mean.item()
            sim_pixel_error_std += pixel_error_std.item()

            pixel_errors_scatter +=[p.item() for p in pixel_error_unormalised]
            pixel_labels_scatter +=[p.item() for p in torch.norm(pixel_deltas_labels, dim = 1)]

        counter_val_loss += outputs.shape[0]
        validation_batch_counter += 1

    res['validation_loss'] = validation_loss / validation_batch_counter

    res['sim_pixel_error'] = sim_pixel_error / validation_batch_counter
    res['sim_scale_error'] = sim_scale_error / validation_batch_counter
    res['sim_angle_error'] = sim_angle_error / validation_batch_counter

    test_direct_reg.validation_loss.append(res['validation_loss'])
    test_direct_reg.sim_pixel_error.append(res['sim_pixel_error'])
    test_direct_reg.sim_pixel_error_std.append(sim_pixel_error_std / validation_batch_counter)
    test_direct_reg.sim_scale_error.append(res['sim_scale_error'])
    test_direct_reg.sim_scale_error_std.append( sim_scale_error_std / validation_batch_counter)
    test_direct_reg.sim_angle_error.append(res['sim_angle_error'])
    test_direct_reg.sim_angle_error_std.append(sim_angle_error / validation_batch_counter)



    plt.figure()
    plt.title('Angle Error')
    plt.xlabel('Epoch number')
    plt.ylabel('angle errors')
    mean_angle_error = np.array(test_direct_reg.sim_angle_error)
    mean_angle_error_std = np.array(test_direct_reg.sim_angle_error_std)
    plt.plot(test_direct_reg.sim_angle_error, label='angle error %')
    plt.fill_between(np.arange(len(mean_angle_error)),mean_angle_error-mean_angle_error_std, mean_angle_error+mean_angle_error_std, alpha = 0.3)

    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, test_direct_reg.save_name, 'Val angle error.png'))
    plt.close()

    plt.figure()
    plt.title('Scale Error')
    plt.xlabel('Epoch number')
    plt.ylabel('scale errors')

    mean_scale_error = np.array(test_direct_reg.sim_scale_error)
    mean_scale_error_std = np.array(test_direct_reg.sim_scale_error_std)
    plt.plot(test_direct_reg.sim_scale_error, label='scale error %')
    plt.fill_between(np.arange(len(mean_scale_error)), mean_scale_error-mean_scale_error_std, mean_scale_error+mean_scale_error_std, alpha =0.3)
    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, test_direct_reg.save_name, 'Val scale error.png'))
    plt.close()

    plt.figure()
    plt.title('Pixel Error')
    plt.xlabel('Epoch number')
    plt.ylabel('pixel errors')

    mean_pixel_error = np.array(test_direct_reg.sim_pixel_error)
    mean_pixel_error_std = np.array(test_direct_reg.sim_pixel_error_std)

    plt.plot(test_direct_reg.sim_pixel_error, label='pixel error %')
    plt.fill_between(np.arange(len(mean_pixel_error)), mean_pixel_error-mean_pixel_error_std,mean_pixel_error+mean_pixel_error_std,alpha = 0.3 )
    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, test_direct_reg.save_name, 'Val pixel error.png'))
    plt.close()

    plt.figure()
    plt.title('Angle Error Scatter')
    plt.xlabel('Angle label')
    plt.ylabel('Angle errors')
    plt.scatter(angles_label_scatter, angles_error_scatter)
    m = np.mean(angles_error_scatter)
    plt.axhline(y=m,label = 'mean: {}'.format(m))
    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, test_direct_reg.save_name, 'Val angle error scatter.png'))
    plt.close()

    plt.figure()
    plt.title('Scale Error Scatter')
    plt.xlabel('scale label')
    plt.ylabel('scale errors')
    plt.scatter(scales_label_scatter, scales_error_scatter)
    m = np.mean(scales_error_scatter)
    plt.axhline(y=m,label = 'mean: {}'.format(m))
    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, test_direct_reg.save_name, 'Val scale error scatter.png'))
    plt.close()



    plt.figure()
    plt.title('Pixel Error Scatter')
    plt.xlabel('pixel label')
    plt.ylabel('pixel errors')
    plt.scatter(pixel_labels_scatter, pixel_errors_scatter)
    m = np.mean(pixel_errors_scatter)
    plt.axhline(y=m,label = 'mean : {}'.format(m))
    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, test_direct_reg.save_name, 'Val pixel error scatter.png'))
    plt.close()










    return res

test_direct_reg.validation_loss = []
test_direct_reg.sim_pixel_error = []
test_direct_reg.sim_pixel_error_std = []
test_direct_reg.sim_scale_error = []
test_direct_reg.sim_scale_error_std = []
test_direct_reg.sim_angle_error = []
test_direct_reg.sim_angle_error_std = []