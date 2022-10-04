import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from dense_correspondence_control.utils.globals import MODELS_PATH
from dense_correspondence_control.utils.learning_utilities import min_max_stardardisation_to_01
import cv2
from PIL import Image

def get_segmentation(network, image_rgb_tensor, image_bottleneck_tensor):
    output, logits = network['model'](image_rgb_tensor, image_bottleneck_tensor)

    mask = output > 0.5
    return mask


def test_seg(network_propagator, dataloader, **kwargs):
    device =kwargs['device']
    res = {}
    if(not os.path.exists(kwargs['sim_save_path'])):
        os.makedirs(kwargs['sim_save_path'])
        os.makedirs(kwargs['hard_real_save_path'])
        os.makedirs( kwargs['cap_real_save_path'])
        os.makedirs( kwargs['eraser_real_save_path'])
        os.makedirs( kwargs['scoop_save_path'])
        os.makedirs( kwargs['insert_ring_save_path'])
        os.makedirs( kwargs['place_cup_save_path'])
        os.makedirs( kwargs['writing_save_path'])
        os.makedirs( kwargs['hammer_save_path'])

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
            final_image = np.clip(np.expand_dims(seg_pred, axis=-1) * alpha * [1., 0., 1.] + rgb * (1 - alpha),  a_min=0., a_max=1.)
            final_image = np.concatenate([final_image, samples['bottleneck_rgb'][0].detach().permute(1, 2, 0).cpu().numpy()], axis=0)

            plt.imsave(os.path.join(kwargs['sim_save_path'], '{}.png'.format(i)), final_image)
            plt.close()


        counter_val_loss += seg_labels.shape[0]
        validation_batch_counter += 1



    res['validation_loss'] = validation_loss / validation_batch_counter
    res['sim_seg_iou'] = sim_iou / validation_batch_counter

    test_seg.validation_loss.append(res['validation_loss'])
    test_seg.sim_seg_iou.append(res['sim_seg_iou'])

    ### Real Tests
    hard_test_set_path = kwargs['hard_test_set_path']
    cap_test_set_path = kwargs['cap_test_set_path']
    eraser_test_set_path = kwargs['eraser_test_set_path']

    scoop_test_set_path = kwargs['scoop_test_set_path']
    insert_test_set_path = kwargs['insert_test_set_path']
    place_test_set_path = kwargs['place_cup_set_path']
    writing_test_set_path = kwargs['writing_set_path']
    hammer_test_set_path = kwargs['hammer_set_path']



    hard_iou = 0
    cap_iou = 0
    eraser_iou = 0


    scoop_iou = 0
    insert_iou = 0
    place_iou = 0
    writing_iou = 0
    hammer_iou = 0

    for data_idx, data_path in enumerate([ hard_test_set_path, cap_test_set_path,  eraser_test_set_path, scoop_test_set_path ,insert_test_set_path,place_test_set_path, writing_test_set_path, hammer_test_set_path]):
        inputs_path = os.path.join(data_path,'processed_images')
        labels_path = os.path.join(data_path, 'processed_seg_images')

        bottleneck_rgb = cv2.cvtColor(cv2.imread(os.path.join(inputs_path,'0.png'),-1), cv2.COLOR_BGR2RGB)
        bottleneck_rgb =Image.fromarray(bottleneck_rgb)
        bottleneck_rgb = bottleneck_rgb.resize((64,64))
        bottleneck_rgb= np.array(bottleneck_rgb)
        image_bottleneck_tensor = torch.tensor(copy.deepcopy(bottleneck_rgb) / 255.).to(device).float().permute(2, 0, 1).unsqueeze(0)


        if('first_stage_network' in kwargs):
            bottleneck_initial_segmentation = get_segmentation(kwargs['first_stage_network'], image_bottleneck_tensor, image_bottleneck_tensor)



        for i in range(10):
            image_rgb = cv2.cvtColor(cv2.imread(os.path.join(inputs_path,'{}.png'.format(i)),-1), cv2.COLOR_BGR2RGB)
            image_rgb = Image.fromarray(image_rgb)
            image_rgb = image_rgb.resize((64,64))
            image_rgb = np.array(image_rgb)

            seg_label = cv2.imread(os.path.join(labels_path,'{}.png'.format(i)),-1)
            seg_label = cv2.resize(seg_label,(64,64))

            image_rgb_tensor = torch.tensor(copy.deepcopy(image_rgb)/255.).to(device).float().permute(2,0,1).unsqueeze(0)

            if ('first_stage_network' in kwargs):
                initial_segmentation = get_segmentation(kwargs['first_stage_network'], image_rgb_tensor, image_bottleneck_tensor)
                bottleneck_image_input = image_bottleneck_tensor * bottleneck_initial_segmentation
                current_image_input = image_rgb_tensor*initial_segmentation

                seg_prediction = network_propagator.real_image_testing_forward(bottleneck_image_input, current_image_input)
            else:

                seg_prediction = network_propagator.real_image_testing_forward(image_bottleneck_tensor,image_rgb_tensor)

            iou = np.logical_and(seg_prediction, seg_label).sum() / np.logical_or(seg_prediction, seg_label).sum()  #

            alpha = 0.4
            final_image = np.clip(np.expand_dims(seg_prediction, axis=-1) * alpha * [1., 0., 1.] + image_rgb/255. * (1 - alpha), a_min=0., a_max=1.)
            final_image = np.concatenate([final_image,bottleneck_rgb/255.], axis = 0 )


            if(data_idx == 0):
                hard_iou +=iou
                plt.imsave(os.path.join(kwargs['hard_real_save_path'], '{}.png'.format(i)), final_image)
            elif(data_idx == 1):
                cap_iou +=iou
                plt.imsave(os.path.join(kwargs['cap_real_save_path'], '{}.png'.format(i)), final_image)
            elif(data_idx == 2):
                eraser_iou +=iou
                plt.imsave(os.path.join(kwargs['eraser_real_save_path'], '{}.png'.format(i)), final_image)
            elif(data_idx == 3):
                scoop_iou +=iou
                plt.imsave(os.path.join(kwargs['scoop_save_path'], '{}.png'.format(i)), final_image)
            elif(data_idx == 4):
                insert_iou +=iou
                plt.imsave(os.path.join(kwargs['insert_ring_save_path'], '{}.png'.format(i)), final_image)
            elif(data_idx == 5):
                place_iou +=iou
                plt.imsave(os.path.join(kwargs['place_cup_save_path'], '{}.png'.format(i)), final_image)
            elif(data_idx == 6):
                writing_iou +=iou
                plt.imsave(os.path.join(kwargs['writing_save_path'], '{}.png'.format(i)), final_image)
            elif(data_idx == 7):
                hammer_iou +=iou
                plt.imsave(os.path.join(kwargs['hammer_save_path'], '{}.png'.format(i)), final_image)

            plt.close()




    hard_iou/=10
    cap_iou/=10
    eraser_iou/=10

    scoop_iou /= 10
    insert_iou /= 10
    place_iou /= 10
    writing_iou /= 10
    hammer_iou /= 10



    res['real_hard_iou'] = hard_iou
    res['real_cap_iou'] = cap_iou
    res['real_eraser_iou'] = eraser_iou


    res['scoop_iou'] = scoop_iou
    res['insert_iou'] = insert_iou
    res['place_iou'] = place_iou
    res['writing_iou'] = writing_iou
    res['hammer_iou'] = hammer_iou

    test_seg.real_hard_iou.append(res['real_hard_iou'])
    test_seg.real_cap_iou.append(res['real_cap_iou'])
    test_seg.real_eraser_iou.append(res['real_eraser_iou'])

    test_seg.real_scoop_iou.append(res['scoop_iou'])
    test_seg.real_insert_iou.append(res['insert_iou'])
    test_seg.real_place_iou.append(res['place_iou'])
    test_seg.real_writing_iou.append(res['writing_iou'])
    test_seg.real_hammer_iou.append(res['hammer_iou'])



    plt.figure()
    plt.title('IoU')
    plt.xlabel('Epoch number')
    plt.ylabel('IOU values')
    plt.plot(test_seg.sim_seg_iou, label='sim IoU')
    plt.plot(test_seg.real_hard_iou, label='real hard IoU')
    plt.plot(test_seg.real_cap_iou, label='real cap IoU')
    plt.plot(test_seg.real_eraser_iou, label='real eraser IoU')

    plt.plot(test_seg.real_scoop_iou, label='real scoop IoU')
    plt.plot(test_seg.real_insert_iou, label='real insert IoU')
    plt.plot(test_seg.real_place_iou, label='real place IoU')
    plt.plot(test_seg.real_writing_iou, label='real writing IoU')
    plt.plot(test_seg.real_hammer_iou, label='real hammer IoU')

    plt.grid(True, axis ='y')
    plt.ylim((0.,1.))
    plt.yticks(np.arange(0.,1.,step=0.1))
    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, test_seg.save_name, 'IoU breakdown.png'))
    plt.close()






    return res

test_seg.validation_loss = []
test_seg.sim_seg_iou = []
test_seg.real_hard_iou = []
test_seg.real_cap_iou = []
test_seg.real_eraser_iou = []


test_seg.real_scoop_iou = []
test_seg.real_insert_iou = []
test_seg.real_place_iou = []
test_seg.real_writing_iou = []
test_seg.real_hammer_iou = []
