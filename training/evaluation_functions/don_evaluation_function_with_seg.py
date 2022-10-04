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
    validation_don_loss = 0.
    validation_seg_loss = 0.

    average_distance_correct = 0
    average_distance_wrong  = 0
    average_distance_object_wrong = 0

    std_distance_correct = 0
    std_distance_wrong = 0
    std_distance_object_wrong = 0

    for i, sample in enumerate(dataloader):
        seg_from = sample['seg_from']
        seg_to = sample['seg_to']
        rgb_from = sample['rgb_from']
        rgb_to = sample['rgb_to']
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


        loss, don_loss, seg_loss,  don_loss_correct, don_loss_o_wrong, don_loss_wrong, out_from, out_to, seg_out_from, seg_out_to = network_propagator.testing_forward(sample)


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


            # Seg Saving

            if not os.path.exists(kwargs['seg_save_path']):
                os.makedirs(kwargs['seg_save_path'])

            from_image = rgb_from[0].detach().permute(1, 2, 0).cpu().numpy()
            from_image = min_max_stardardisation_to_01(from_image)
            #from_image = (from_image * 255).astype('uint8')
            to_image = rgb_to[0].detach().permute(1, 2, 0).cpu().numpy()
            to_image = min_max_stardardisation_to_01(to_image)
            #to_image = (to_image * 255).astype('uint8')


            seg_from_image = seg_out_from[0,0].cpu().numpy()
            seg_to_image = seg_out_to[0,0].cpu().numpy()

            alpha = 0.4
            final_from_image = np.clip(np.expand_dims(seg_from_image, axis=-1) * alpha * [1., 0., 0.] + from_image * (1 - alpha),  a_min=0., a_max=1.)
            final_to_image = np.clip(np.expand_dims(seg_to_image, axis=-1) * alpha * [1., 0., 0.] + to_image * (1 - alpha),a_min=0., a_max=1.)


            plt.imsave(os.path.join(kwargs['seg_save_path'], 'epoch_{}_from_{}.png'.format(network_propagator.epoch, i)), final_from_image)
            plt.imsave(os.path.join(kwargs['seg_save_path'], 'epoch{}_to_{}.png'.format(network_propagator.epoch, i)), final_to_image)
            plt.close()
            del final_from_image
            del final_to_image
            del from_image
            del to_image
            del seg_from_image
            del seg_to_image

        validation_loss += loss
        validation_seg_loss += seg_loss
        validation_don_loss += don_loss
        validation_c_loss += don_loss_correct
        validation_o_w_loss += don_loss_o_wrong
        validation_w_loss += don_loss_wrong

        counter_val_loss += sample['rgb_from'].shape[0]
        validation_batch_counter += 1

    res['validation_loss'] = validation_loss / validation_batch_counter
    res['validation_seg_loss'] = validation_seg_loss / validation_batch_counter
    res['validation_don_loss'] = validation_don_loss / validation_batch_counter
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
    test_don.validation_seg_loss.append(res['validation_seg_loss'])
    test_don.validation_don_loss.append(res['validation_don_loss'])
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
    plt.plot(np.log(test_don.validation_seg_loss), label='Val Seg Loss')
    plt.plot(np.log(test_don.validation_don_loss), label='Val Don Loss')

    plt.locator_params(axis="y", nbins=30)

    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, test_don.save_name, 'val_losses_breakdown.png'))
    plt.close()


    plt.figure()
    plt.title('Validation_losses')
    plt.xlabel('Epoch number')
    plt.ylabel('LOG loss_values')
    plt.plot(np.log(test_don.validation_c_loss), label='Val Correct Loss')
    plt.plot(np.log(test_don.validation_o_w_loss), label='Val Object Wrong Loss')
    plt.plot(np.log(test_don.validation_w_loss), label='Val Wrong Loss')

    plt.locator_params(axis="y", nbins=30)

    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, test_don.save_name, 'val_don_losses_breakdown.png'))
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

    plt.locator_params(axis="y", nbins=30)

    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, test_don.save_name, 'val_dist_breakdown.png'))
    plt.close()

    return res



test_don.validation_loss = []
test_don.validation_seg_loss = []
test_don.validation_don_loss = []
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

    loss_correct = info['don_loss_correct']
    loss_o_wrong = info['don_loss_o_wrong']
    loss_wrong = info['don_loss_wrong']
    out_from = info['out_from']
    out_to = info['out_to']

    seg_loss =  info['seg_loss']
    don_loss = info['don_loss']


    num_batches = len(don_in_training_evaluation.loss_correct) + 1

    prev_loss_correct = don_in_training_evaluation.loss_correct[-1] if num_batches > 1 else 0
    don_in_training_evaluation.loss_correct.append(prev_loss_correct + (loss_correct - prev_loss_correct) / num_batches)

    prev_loss_o_wrong = don_in_training_evaluation.loss_o_wrong[-1] if num_batches > 1 else 0
    don_in_training_evaluation.loss_o_wrong.append(prev_loss_o_wrong + (loss_o_wrong - prev_loss_o_wrong) / num_batches)

    prev_loss_wrong = don_in_training_evaluation.loss_wrong[-1] if num_batches > 1 else 0
    don_in_training_evaluation.loss_wrong.append(prev_loss_wrong + (loss_wrong - prev_loss_wrong) / num_batches)

    prev_loss_don = don_in_training_evaluation.don_loss[-1] if num_batches > 1 else 0
    don_in_training_evaluation.don_loss.append(prev_loss_don + (don_loss - prev_loss_don) / num_batches)

    prev_seg_loss = don_in_training_evaluation.seg_loss[-1] if num_batches > 1 else 0
    don_in_training_evaluation.seg_loss.append(prev_seg_loss + (seg_loss - prev_seg_loss) / num_batches)

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

        plt.figure()
        plt.title('In Training DON SEG Losses')
        plt.xlabel('Epoch number')
        plt.ylabel('LOG loss_values')
        plt.plot(np.log(don_in_training_evaluation.don_loss), label='don loss')
        plt.plot(np.log(don_in_training_evaluation.seg_loss), label='seg loss')

        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_base, 'seg_don_losses_breakdown.png'))
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
don_in_training_evaluation.seg_loss = []
don_in_training_evaluation.don_loss = []