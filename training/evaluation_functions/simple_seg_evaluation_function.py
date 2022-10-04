import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from dense_correspondence_control.utils.globals import MODELS_PATH
from dense_correspondence_control.utils.learning_utilities import min_max_stardardisation_to_01
import cv2

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

def test_seg(network_propagator, dataloader, **kwargs):

    res = {}
    if(not os.path.exists(kwargs['sim_save_path'])):
        os.makedirs(kwargs['sim_save_path'])
        os.makedirs(kwargs['easy_real_save_path'])
        os.makedirs(kwargs['hard_real_save_path'])


    ### Sim evaluation
    counter_val_loss = 0
    validation_batch_counter = 0.
    validation_loss = 0.
    sim_iou = 0.


    for i, samples in enumerate(dataloader):
        seg_labels = samples['current_seg']
        loss, seg_preds = network_propagator.testing_forward(samples)

        # accumulate loss
        validation_loss += loss

        # accumulate iou
        np_seg_pred = seg_preds.squeeze().detach().cpu().numpy()
        np_label =seg_labels.squeeze().detach().cpu().numpy()
        sim_iou += np.mean(np.logical_and(np_seg_pred,np_label ).sum(axis=(1,2)) / np.logical_or(np_seg_pred, np_label).sum(axis=(1,2)))

        # saving_idx = np.random.choice(len(dataloader))

        if(i<10):
            seg_pred = seg_preds[0,0].detach().cpu().numpy()
            rgb = samples['current_rgb'][0].detach().permute(1, 2, 0).cpu().numpy()
            alpha = 0.4
            final_image = np.clip(np.expand_dims(seg_pred, axis=-1) * alpha * [1., 0., 0.] + rgb * (1 - alpha),  a_min=0., a_max=1.)

            plt.imsave(os.path.join(kwargs['sim_save_path'], '{}.png'.format(i)), final_image)
            plt.close()


        counter_val_loss += seg_labels.shape[0]
        validation_batch_counter += 1

    del final_image
    del rgb
    del seg_pred

    res['validation_loss'] = validation_loss / validation_batch_counter
    res['sim_seg_iou'] = sim_iou / validation_batch_counter

    test_seg.validation_loss.append(res['validation_loss'])
    test_seg.sim_seg_iou.append(res['sim_seg_iou'])

    ### Real Tests

    easy_test_set_path = kwargs['easy_test_set_path']
    hard_test_set_path = kwargs['hard_test_set_path']

    easy_iou  = 0
    hard_iou = 0

    for data_idx, data_path in enumerate([easy_test_set_path, hard_test_set_path]):
        inputs_path = os.path.join(data_path,'processed_images')
        labels_path = os.path.join(data_path, 'processed_seg_images')

        for i in range(1,20):
            image_rgb = cv2.cvtColor(cv2.imread(os.path.join(inputs_path,'{}.png'.format(i)),-1), cv2.COLOR_BGR2RGB)
            seg_label = cv2.imread(os.path.join(labels_path,'{}.png'.format(i)),-1)

            image_rgb_tensor = torch.tensor(copy.deepcopy(image_rgb)/255.).to(device).float().permute(2,0,1).unsqueeze(0)

            seg_prediction = network_propagator.real_image_testing_forward(image_rgb_tensor)

            iou = np.logical_and(seg_prediction, seg_label).sum() / np.logical_or(seg_prediction, seg_label).sum()  #

            alpha = 0.4
            final_image = np.clip(np.expand_dims(seg_prediction, axis=-1) * alpha * [1., 0., 0.] + image_rgb/255. * (1 - alpha), a_min=0., a_max=1.)

            if(data_idx <1):
                easy_iou+=iou
                plt.imsave(os.path.join(kwargs['easy_real_save_path'], '{}.png'.format(i)), final_image)
            else:
                hard_iou +=iou
                plt.imsave(os.path.join(kwargs['hard_real_save_path'], '{}.png'.format(i)), final_image)

            plt.close()

    del final_image
    del image_rgb
    del seg_prediction

    easy_iou/= 19
    hard_iou/=19
    res['real_easy_iou'] = easy_iou
    res['real_hard_iou'] = hard_iou

    test_seg.real_easy_iou.append(res['real_easy_iou'])
    test_seg.real_hard_iou.append(res['real_hard_iou'])


    plt.figure()
    plt.title('IoU')
    plt.xlabel('Epoch number')
    plt.ylabel('IOU values')
    plt.plot(test_seg.sim_seg_iou, label='sim IoU')
    plt.plot(test_seg.real_easy_iou, label='real easy IoU')
    plt.plot(test_seg.real_hard_iou, label='real hard IoU')
    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, test_seg.save_name, 'IoU breakdown.png'))

    return res

test_seg.validation_loss = []
test_seg.sim_seg_iou = []
test_seg.real_easy_iou = []
test_seg.real_hard_iou = []