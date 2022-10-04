
from dense_correspondence_control.utils.globals import *
import glob
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import PIL
import PIL.ImageFilter as filter
import pickle

from dense_correspondence_control.learning.data_processing.apply_postprocessing import  postprocess_image
from dense_correspondence_control.learning.data_processing.apply_background import get_textures,apply_background_with_images
import shutil
from dense_correspondence_control.utils.image_utils import calculate_iou
from dense_correspondence_control.learning.testing.load_networks import load_network
import torch


class GetSegFromNetworks(object):
    def __init__(self):
        # Load network
        self.device = torch.device('cuda:0')
        networks, _ = load_network(
            '/home/eugene/Projects/dense_correspondence_control/dense_correspondence_control/learning/data/models/segmentations/seg_w_distractors_no_bg_larger2',
            185, device=self.device)

        self.networks = networks


    def get_seg_map_networks(self, current_rgb, bottleneck_rgb) :
        # Put image in torch format
        rgb_current = torch.tensor(current_rgb/255.).to(self.device).float().permute(2,0,1).unsqueeze(0)
        rgb_bottleneck = torch.tensor(bottleneck_rgb/255.).to(self.device).float().permute(2,0,1).unsqueeze(0)

        output, logits = self.networks['model'](rgb_current,rgb_bottleneck)
        mask = output > 0.5

        return mask.squeeze().detach().cpu().numpy().astype(float)





if __name__ == '__main__':
    print('PID!!! {}'.format(os.getpid()))
    #set the seed
    seed = 0
    np.random.seed(seed)
    # make the dataset save path
    save_path= os.path.join( DATASETS_PATH, 'small_displacements_direct_regression/' )

    if(not os.path.exists(save_path)):
        os.makedirs(save_path)

    # blenderproc path
    blenderproc_path = '/home/eugene/Projects/BlenderProc/examples/data/datasets/small_displacement_dataset/'
    folder_paths = glob.glob(os.path.join(blenderproc_path,'*/'))

    all_backgrounds = get_textures()


    segmenter = GetSegFromNetworks()



    bottleneck_rgb_data  = []
    bottleneck_seg_from_network_data = []
    bottleneck_object_seg_data  = []

    current_rgb_data  = []
    current_seg_from_network_data = []
    current_object_seg_data  = []

    current_rgb_data_at_bottleneck  = []
    current_seg_from_network_data_at_bottleneck = []
    current_object_seg_data_at_bottleneck  = []

    z_angles = []


    count = 0
    for folder_path in folder_paths:
        folder_name = folder_path.split('/')[-2]
        print('Doing Folder {}'.format(folder_name))

        bottleneck_z_in_current_path = os.path.join(folder_path, 'bottleneck_z_angles_in_current_frame.pckl')
        if(not os.path.exists(bottleneck_z_in_current_path)):
            print('no bottleneck_z_in_current_path, skipping')
            continue

        bottleneck_z_in_current = pickle.load(open(bottleneck_z_in_current_path, 'rb'))

        bottleneck_image_paths = glob.glob(os.path.join(folder_path, 'bottleneck_rgb/*.png'))
        for i in range(len(bottleneck_image_paths)): #
            bottleneck_image_path = os.path.join(folder_path,'bottleneck_rgb','{:06d}.png'.format(i))
            save_bottleneck_seg_from_network_path = os.path.join(folder_path,'bottleneck_seg_from_network','{:06d}.png'.format(i))
            bottleneck_object_seg_path = os.path.join(folder_path, 'bottleneck_object_seg', '{:06d}.png'.format(i))

            bottleneck_rgb = cv2.cvtColor(cv2.imread(bottleneck_image_path), cv2.COLOR_BGR2RGB)

            bottleneck_seg_image_from_network = segmenter.get_seg_map_networks(bottleneck_rgb, bottleneck_rgb)

            if (not os.path.exists(save_bottleneck_seg_from_network_path)):
                if (not os.path.exists(os.path.dirname(save_bottleneck_seg_from_network_path))):
                    os.makedirs(os.path.dirname(save_bottleneck_seg_from_network_path))

            cv2.imwrite(save_bottleneck_seg_from_network_path, bottleneck_seg_image_from_network)

            skip = False
            for f_name in ['current_rgb', 'current_object_seg', 'current_distractor_seg', 'current_normals', 'current_distance', 'current_depth']:

                f_p = glob.glob(os.path.join(folder_path, '{}/*.png').format(f_name))
                if(len(f_p) <3 ):
                    print('No current images, skipping, folder {}'.format(folder_name))
                    skip= True
                    break

            if(skip):
                continue





            current_image_path_at_bottleneck = os.path.join(folder_path, 'current_rgb', '000000.png')
            current_save_seg_from_network_path_at_bottleneck = os.path.join(folder_path, 'current_seg_from_network','000000.png')
            current_object_seg_path_at_bottleneck = os.path.join(folder_path, 'current_object_seg', '000000.png')

            if (i == 0):
                if (not os.path.exists(os.path.dirname(current_save_seg_from_network_path_at_bottleneck))):
                    os.makedirs(os.path.dirname(current_save_seg_from_network_path_at_bottleneck))

                current_rgb = cv2.cvtColor(cv2.imread(current_image_path_at_bottleneck), cv2.COLOR_BGR2RGB)
                current_seg_image_from_network = segmenter.get_seg_map_networks(current_rgb, bottleneck_rgb)
                cv2.imwrite(current_save_seg_from_network_path_at_bottleneck, current_seg_image_from_network)



            current_image_paths = glob.glob(os.path.join(folder_path, 'current_rgb/*.png').format(f_name))

            for j in range(1, np.minimum(3,len(current_image_paths))):
                count +=1
                # j=j+1
                current_image_path = os.path.join(folder_path, 'current_rgb', '{:06d}.png'.format(j))
                current_save_seg_from_network_path = os.path.join(folder_path, 'current_seg_from_network', '{:06d}.png'.format(j))
                current_object_seg_path = os.path.join(folder_path, 'current_object_seg', '{:06d}.png'.format(j))


                if(i==0):
                    current_rgb = cv2.cvtColor(cv2.imread(current_image_path), cv2.COLOR_BGR2RGB )
                    current_seg_image_from_network = segmenter.get_seg_map_networks(current_rgb, bottleneck_rgb)
                    cv2.imwrite(current_save_seg_from_network_path,current_seg_image_from_network)


                bottleneck_rgb_data.append(bottleneck_image_path)
                bottleneck_seg_from_network_data.append(save_bottleneck_seg_from_network_path)
                bottleneck_object_seg_data.append(bottleneck_object_seg_path)

                current_rgb_data.append(current_image_path)
                current_seg_from_network_data.append(current_save_seg_from_network_path)
                current_object_seg_data.append(current_object_seg_path)

                current_rgb_data_at_bottleneck.append(current_image_path_at_bottleneck)
                current_seg_from_network_data_at_bottleneck.append(current_object_seg_path_at_bottleneck)
                current_object_seg_data_at_bottleneck.append(current_object_seg_path_at_bottleneck)

                ## Add to the regression arrays
                z_angles.append(bottleneck_z_in_current['{:06d}_{:06d}'.format(j,i)][2])

                print('Done {}'.format(count))

    print('Done')
    print('dataset_size : {}'.format(len(current_rgb_data)))


    pickle.dump(bottleneck_rgb_data,open(os.path.join(save_path,'bottleneck_rgb_data_paths.pckl'), 'wb'))
    pickle.dump(bottleneck_seg_from_network_data,open(os.path.join(save_path,'bottleneck_seg_from_network_data_paths.pckl'), 'wb'))
    pickle.dump(bottleneck_object_seg_data,open(os.path.join(save_path,'bottleneck_object_seg_data_paths.pckl'), 'wb'))

    pickle.dump(current_rgb_data,open(os.path.join(save_path,'current_rgb_data_paths.pckl'), 'wb'))
    pickle.dump(current_seg_from_network_data,  open(os.path.join(save_path,'current_seg_from_network_data_paths.pckl'), 'wb')   )
    pickle.dump(current_object_seg_data,open(os.path.join(save_path,'current_object_seg_data_paths.pckl'), 'wb'))


    pickle.dump(current_rgb_data_at_bottleneck,open(os.path.join(save_path,'current_rgb_data_paths_at_bottleneck.pckl'), 'wb'))
    pickle.dump(current_seg_from_network_data_at_bottleneck,  open(os.path.join(save_path,'current_seg_from_network_data_paths_at_bottleneck.pckl'), 'wb')   )
    pickle.dump(current_object_seg_data_at_bottleneck,open(os.path.join(save_path,'current_object_seg_data_paths_at_bottleneck.pckl'), 'wb'))


    z_angles_array = np.array(z_angles)

    np.save(os.path.join(save_path, 'z_angles.npy'), z_angles_array)
    print('DONE')

