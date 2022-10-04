import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from dense_correspondence_control.utils.globals import MODELS_PATH
from dense_correspondence_control.utils.learning_utilities import min_max_stardardisation_to_01




def test_don(network_propagator, dataloader, **kwargs):
    res = {}

    counter_val_loss = 0
    validation_batch_counter = 0.
    validation_loss = 0.
    validation_c_loss = 0.
    validation_o_w_loss = 0.
    validation_w_loss = 0.

    average_distance_correct = 0
    average_distance_wrong  = 0
    average_distance_object_wrong = 0

    std_distance_correct = 0
    std_distance_wrong = 0
    std_distance_object_wrong = 0

    for i, sample in enumerate(dataloader):
        from_x = sample['from_x']
        from_y = sample['from_y']
        to_x = sample['to_x']
        to_y = sample['to_y']
        from_w_x = sample['from_w_x']
        from_w_y = sample['from_w_y']
        to_w_x = sample['to_w_x']
        to_w_y = sample['to_w_y']
        from_w_o_x = sample['from_w_o_x']
        from_w_o_y = sample['from_w_o_y']
        to_w_o_x = sample['to_w_o_x']
        to_w_o_y = sample['to_w_o_y']


        loss, loss_correct, loss_o_wrong, loss_wrong, out_from, out_to = network_propagator.testing_forward(sample)


        ## Find the average distance of representaitons in the different categories ####

        outs_from_correct = torch.empty((out_from.shape[0], out_from.shape[1], from_x.shape[1]))
        for j in range(out_to.shape[0]):
            outs_from_correct[j] = out_from[j, :, from_y[j, :], from_x[j, :]]

        outs_to_correct = torch.empty((out_to.shape[0], out_to.shape[1], from_x.shape[1]))
        for j in range(out_to.shape[0]):
            outs_to_correct[j] = out_to[j, :, to_y[j, :], to_x[j, :]]

        distances_correct = np.linalg.norm(outs_from_correct.detach().cpu().numpy()- outs_to_correct.detach().cpu().numpy(), axis = 1)
        average_distance_correct += np.mean(distances_correct)
        std_distance_correct += np.std(distances_correct)



        outs_from_wrong = torch.empty((out_to.shape[0],  out_from.shape[1], from_w_y.shape[1]))
        for j in range(out_to.shape[0]):
            outs_from_wrong[j] = out_from[j, :, from_w_y[j,:], from_w_x[j,:]]

        outs_to_wrong = torch.empty((out_to.shape[0], out_from.shape[1], from_w_y.shape[1]))
        for j in range(out_to.shape[0]):
            outs_to_wrong[j] = out_to[j, :, to_w_y[j,:], to_w_x[j,:]]

        distances_wrong = np.linalg.norm(outs_from_wrong.detach().cpu().numpy() - outs_to_wrong.detach().cpu().numpy(), axis = 1)
        average_distance_wrong += np.mean(distances_wrong)
        std_distance_wrong += np.std(distances_wrong)

        outs_from_obj_wrong = torch.empty((out_to.shape[0], out_from.shape[1], from_w_o_x.shape[1]))
        for j in range(out_to.shape[0]):
            outs_from_obj_wrong[j] = out_from[j, :, from_w_o_y[j, :], from_w_o_x[j, :]]

        outs_to_obj_wrong = torch.empty((out_to.shape[0], out_from.shape[1], from_w_o_x.shape[1]))
        for j in range(out_to.shape[0]):
            outs_to_obj_wrong[j] = out_from[j, :, to_w_o_y[j, :], to_w_o_x[j, :]]

        distances_wo = np.linalg.norm(outs_from_obj_wrong.detach().cpu().numpy()- outs_to_obj_wrong.detach().cpu().numpy(), axis = 1)
        average_distance_object_wrong += np.mean(distances_wo)
        std_distance_object_wrong += np.std(distances_wo)

        if(i<1) :

            # DONS saving
            if not os.path.exists(kwargs['image_save_path']):
                os.makedirs(kwargs['image_save_path'])


            from_image = out_from[0].permute(1,2,0).cpu().numpy()
            from_image = min_max_stardardisation_to_01(from_image)
            from_image = (from_image*255).astype('uint8')
            to_image = out_to[0].permute(1,2,0).cpu().numpy()
            to_image = min_max_stardardisation_to_01(to_image)
            to_image =  (to_image*255).astype('uint8')

            plt.imsave(os.path.join(kwargs['image_save_path'], 'epoch_{}_from_{}.png'.format(network_propagator.epoch, i)), from_image)
            plt.imsave(os.path.join(kwargs['image_save_path'], 'epoch{}_to_{}.png'.format(network_propagator.epoch, i)), to_image)
            plt.close()


        validation_loss += loss
        validation_c_loss += loss_correct
        validation_o_w_loss += loss_o_wrong
        validation_w_loss += loss_wrong

        counter_val_loss += sample['rgb_from'].shape[0]
        validation_batch_counter += 1

    res['validation_loss'] = validation_loss / validation_batch_counter
    res['validation_c_loss'] = validation_c_loss / validation_batch_counter
    res['validation_o_w_loss'] = validation_o_w_loss / validation_batch_counter
    res['validation_w_loss'] = validation_w_loss / validation_batch_counter


    res['validation_correct_dist'] = average_distance_correct / validation_batch_counter
    res['validation_o_w_dist'] = average_distance_object_wrong / validation_batch_counter
    res['validation_w_dist'] = average_distance_wrong / validation_batch_counter

    res['validation_std_correct_dist'] = std_distance_correct / validation_batch_counter
    res['validation_std_o_w_dist'] = std_distance_object_wrong / validation_batch_counter
    res['validation_std_w_dist'] = std_distance_wrong / validation_batch_counter


    test_don.validation_loss.append(res['validation_loss'])
    test_don.validation_c_loss.append(res['validation_c_loss'])
    test_don.validation_o_w_loss.append(res['validation_o_w_loss'])
    test_don.validation_w_loss.append(res['validation_w_loss'])


    test_don.validation_c_dist.append(res['validation_correct_dist'])
    test_don.validation_o_w_dist.append(res['validation_o_w_dist'])
    test_don.validation_w_dist.append(res['validation_w_dist'])


    test_don.validation_std_correct_dist.append(res['validation_std_correct_dist'])
    test_don.validation_std_o_w_dist.append(res['validation_std_o_w_dist'])
    test_don.validation_std_w_dist.append(res['validation_std_w_dist'])



    plt.figure()
    plt.title('Validation_losses')
    plt.xlabel('Epoch number')
    plt.ylabel('LOG loss_values')
    plt.plot(np.log(test_don.validation_loss), label='Val Loss')
    plt.plot(np.log(test_don.validation_c_loss), label='Val Correct Loss')
    plt.plot(np.log(test_don.validation_o_w_loss), label='Val Object Wrong Loss')
    plt.plot(np.log(test_don.validation_w_loss), label='Val Wrong Loss')

    plt.locator_params(axis="y", nbins=20)

    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, test_don.save_name, 'val_losses_breakdown.png'))
    plt.close()

    plt.figure()
    plt.title('Validation_Distances')
    plt.xlabel('Epoch number')
    plt.ylabel('Distance values')
    plt.plot(test_don.validation_c_dist, label='Val Correct Dist')
    plt.fill_between(np.arange(len(test_don.validation_c_dist)),np.array(test_don.validation_c_dist) -np.array(test_don.validation_std_correct_dist), np.array(test_don.validation_c_dist) +np.array(test_don.validation_std_correct_dist), alpha =0.2, lw=0)
    plt.plot(test_don.validation_o_w_dist, label='Val Object Wrong Dist')
    plt.fill_between(np.arange(len(test_don.validation_o_w_dist)),np.array(test_don.validation_o_w_dist) -np.array(test_don.validation_std_o_w_dist), np.array(test_don.validation_o_w_dist) +np.array(test_don.validation_std_o_w_dist), alpha =0.2, lw=0)

    plt.plot(test_don.validation_w_dist, label='Val Wrong Dist')
    plt.fill_between(np.arange(len(test_don.validation_w_dist)),np.array(test_don.validation_w_dist) -np.array(test_don.validation_std_w_dist), np.array(test_don.validation_w_dist) +np.array(test_don.validation_std_w_dist), alpha =0.2, lw=0)

    plt.locator_params(axis="y", nbins=20)

    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, test_don.save_name, 'val_dist_breakdown.png'))
    plt.close()

    return res



test_don.validation_loss = []
test_don.validation_c_loss = []
test_don.validation_o_w_loss = []
test_don.validation_w_loss = []
test_don.validation_c_dist = []
test_don.validation_o_w_dist = []
test_don.validation_w_dist = []
test_don.validation_std_correct_dist =[]
test_don.validation_std_o_w_dist =[]
test_don.validation_std_w_dist =[]



def don_in_training_evaluation(info):
    don_in_training_evaluation.count += 1

    loss_correct = info['loss_correct']
    loss_o_wrong = info['loss_o_wrong']
    loss_wrong = info['loss_wrong']
    out_from = info['out_from']
    out_to = info['out_to']

    num_batches = len(don_in_training_evaluation.loss_correct) + 1

    prev_loss_correct = don_in_training_evaluation.loss_correct[-1] if num_batches > 1 else 0
    don_in_training_evaluation.loss_correct.append(prev_loss_correct + (loss_correct - prev_loss_correct) / num_batches)

    prev_loss_o_wrong = don_in_training_evaluation.loss_o_wrong[-1] if num_batches > 1 else 0
    don_in_training_evaluation.loss_o_wrong.append(prev_loss_o_wrong + (loss_o_wrong - prev_loss_o_wrong) / num_batches)

    prev_loss_wrong = don_in_training_evaluation.loss_wrong[-1] if num_batches > 1 else 0
    don_in_training_evaluation.loss_wrong.append(prev_loss_wrong + (loss_wrong - prev_loss_wrong) / num_batches)

    if(don_in_training_evaluation.count %100 ==0):
        save_base = os.path.join(MODELS_PATH, test_don.save_name, 'in_training_losses/')
        save_base_image = os.path.join(save_base, 'images')
        if(not os .path.exists(save_base)):
            os.makedirs(save_base)
            os.makedirs(save_base_image)

        plt.figure()
        plt.title('In Training  Losses')
        plt.xlabel('Epoch number')
        plt.ylabel('LOG loss_values')
        plt.plot(np.log(don_in_training_evaluation.loss_correct), label='Correct Loss')
        plt.plot(np.log(don_in_training_evaluation.loss_o_wrong), label='Object Wrong Loss')
        plt.plot(np.log(don_in_training_evaluation.loss_wrong), label='Wrong Loss')

        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_base, 'training_losses_breakdown.png'))
        plt.close()


        i = don_in_training_evaluation.count %5
        from_image = out_from[0].permute(1, 2, 0).cpu().numpy()
        from_image = min_max_stardardisation_to_01(from_image)
        from_image = (from_image * 255).astype('uint8')
        to_image = out_to[0].permute(1, 2, 0).cpu().numpy()
        to_image = min_max_stardardisation_to_01(to_image)
        to_image = (to_image * 255).astype('uint8')

        plt.imsave(os.path.join(save_base_image, 'from_{}.png'.format(i)), from_image)
        plt.imsave(os.path.join(save_base_image, 'to_{}.png'.format(i)), to_image)
        plt.close()




don_in_training_evaluation.count =0
don_in_training_evaluation.loss_correct = []
don_in_training_evaluation.loss_o_wrong = []
don_in_training_evaluation.loss_wrong = []