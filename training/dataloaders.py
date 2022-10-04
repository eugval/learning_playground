import os
import glob
import cv2

import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pickle
import copy


class CorrespondenceDataGrabber(Dataset):
    def __init__(self,
                 low_dim_data_paths=[],
                 low_dim_data_names=[],
                 image_filename_paths= [],
                 image_data_folders=[],
                 image_data_names=[],
                 image_data_types = [],
                 device=None,
                 dataset_size = None,
                 ):
        # Make sure arguments make sense
        if (isinstance(low_dim_data_paths, str)):
            low_dim_data_paths = [low_dim_data_paths]
            low_dim_data_names = [low_dim_data_names]

        if (isinstance(image_data_folders, str)):
            image_data_folders = [image_data_folders]
            image_data_names = [image_data_names]



        assert len(low_dim_data_paths) == len(low_dim_data_names)
        assert len(image_data_folders) == len(image_data_names)
        assert len(image_data_folders) == len(image_data_types)
        assert len(image_data_folders) == len(image_filename_paths)


        if (device is None):
            self.device = torch.device('cpu')
        else:
            self.device = device

        # Load low dim data, transform them and get them to tensor
        self.low_dim_data_paths = low_dim_data_paths
        self.low_dim_data_names = low_dim_data_names

        if(dataset_size is None):
            self.dataset_length = len(glob.glob(os.path.join(low_dim_data_paths[0], '*.pt')))
        else:
            self.dataset_length = dataset_size

        self.image_data_folders = image_data_folders
        self.image_data_names = image_data_names
        self.image_data_types = image_data_types
        self.image_filenames = [torch.load(image_filename_paths[i]) for i in range(len(image_filename_paths))]

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        sample = {}

        for i, name in enumerate(self.low_dim_data_names):
            f_path = self.low_dim_data_paths[i]
            sample[name] = torch.from_numpy(torch.load(os.path.join(f_path,'{}.pt'.format(idx))).squeeze())



        for i, name in enumerate(self.image_data_names):
            f_path = self.image_data_folders[i]
            file_type = self.image_data_types[i]
            filename = self.image_filenames[i][idx]

            image = self.retrieve_image(filename, image_folder =f_path, image_type =file_type)

            sample[name] = torch.tensor(image).to(
                self.device).float().permute(2, 0, 1)

        return sample


    def retrieve_image(self, idx, image_folder, image_type = '.png'):
        image_path = os.path.join(image_folder, '{}{}'.format(idx, image_type))

        image = cv2.imread(image_path, -1).astype(np.float32)

        if (len(image.shape) < 3):
            image = image[:, :, None]

        if (image.shape[2] == 3):
            #TODO: RIGHT NOW IM USING BGR!!!!!!! :
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0

        return image


class SegmentationDataGrabber(Dataset):
    def __init__(self,
                 low_dim_data_paths=[],
                 low_dim_data_names=[],
                 image_data_folders=[],
                 image_data_names=[],
                 image_data_types = [],
                 device=None,
                 dataset_size = None,
                 preloaded_paths = False
                 ):
        # Make sure arguments make sense
        if (isinstance(low_dim_data_paths, str)):
            low_dim_data_paths = [low_dim_data_paths]
            low_dim_data_names = [low_dim_data_names]

        if (isinstance(image_data_folders, str)):
            image_data_folders = [image_data_folders]
            image_data_names = [image_data_names]

        assert len(low_dim_data_paths) == len(low_dim_data_names)
        assert len(image_data_folders) == len(image_data_names)
        assert len(image_data_folders) == len(image_data_types)

        if (device is None):
            self.device = torch.device('cpu')
        else:
            self.device = device

        # Load low dim data, transform them and get them to tensor
        self.low_dim_data_paths = low_dim_data_paths
        self.low_dim_data_names = low_dim_data_names

        if(dataset_size is None):
            self.dataset_length = len(glob.glob(os.path.join(image_data_folders[0], '*.png')))
        else:
            self.dataset_length = dataset_size

        self.image_data_folders = image_data_folders
        self.image_data_names = image_data_names
        self.image_data_types = image_data_types

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        sample = {}

        for i, name in enumerate(self.low_dim_data_names):
            f_path = self.low_dim_data_paths[i]
            sample[name] = torch.from_numpy(torch.load(os.path.join(f_path,'{}.pt'.format(idx))).squeeze())



        for i, name in enumerate(self.image_data_names):
            f_path = self.image_data_folders[i]
            file_type = self.image_data_types[i]
            file_path =os.path.join(f_path,'{}{}'.format(idx,file_type))

            image = self.retrieve_image(file_path)

            sample[name] = torch.tensor(image).to(
                self.device).float().permute(2, 0, 1)

        return sample


    def retrieve_image(self, file_path):
        image = cv2.imread(file_path, -1).astype(np.float32)

        if (len(image.shape) < 3):
            image = image[:, :, None]

        if (image.shape[2] == 3):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0

        return image



class NewSegmentationDataGrabber(Dataset):
    def __init__(self,
                 low_dim_data_paths=[],
                 low_dim_data_names=[],
                 image_data_paths=[],
                 image_data_names=[],
                 device=None,
                 dataset_size=None,
                 index_resuffle = False,
                 ):
        # Make sure arguments make sense
        if (isinstance(low_dim_data_paths, str)):
            low_dim_data_paths = [low_dim_data_paths]
            low_dim_data_names = [low_dim_data_names]

        if (isinstance(image_data_paths, str)):
            image_data_paths = [image_data_paths]
            image_data_names = [image_data_names]

        assert len(low_dim_data_paths) == len(low_dim_data_names)
        assert len(image_data_paths) == len(image_data_names)

        if (device is None):
            self.device = torch.device('cpu')
        else:
            self.device = device

        # Load low dim data, transform them and get them to tensor
        self.low_dim_data_paths = low_dim_data_paths
        self.low_dim_data_names = low_dim_data_names

        self.image_data_paths = {}
        for i,name in enumerate(image_data_names):
            if(isinstance(image_data_paths,str)):
                self.image_data_paths[name] = pickle.load(open(image_data_paths[i],'rb'))
            else:
                self.image_data_paths[name] = image_data_paths[i]


        self.image_data_names = image_data_names


        if (dataset_size is None):
            self.dataset_length = len(self.image_data_paths[self.image_data_names[0]])
        else:
            self.dataset_length = dataset_size


        if(index_resuffle):
            self.index_reshuffling = copy.deepcopy(np.arange(len(self.image_data_paths[self.image_data_names[0]])))
            np.random.shuffle(self.index_reshuffling)
        else:
            self.index_reshuffling = None


    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if(self.index_reshuffling is not None):
            idx = self.index_reshuffling[idx]

        sample = {}

        for i, name in enumerate(self.low_dim_data_names):
            f_path = self.low_dim_data_paths[i]
            sample[name] = torch.from_numpy(torch.load(os.path.join(f_path, '{}.pt'.format(idx))).squeeze())

        for i, name in enumerate(self.image_data_names):
            file_path = self.image_data_paths[name][idx]

            image = self.retrieve_image(file_path)

            sample[name] = torch.tensor(image).to(
                self.device).float().permute(2, 0, 1)

        return sample

    def retrieve_image(self, file_path):
        image = cv2.imread(file_path, -1)
        if(image is None):
            print('FFS')

        image = image.astype(np.float32)

        if (len(image.shape) < 3):
            image = image[:, :, None]

        if (image.shape[2] == 3):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0

        return image



class DirectRegressionDataGrabber(Dataset):
    def __init__(self,
                 low_dim_data=[],
                 low_dim_data_names=[],
                 image_data_paths=[],
                 image_data_names=[],
                 image_selection_names = [],
                 image_selection_probabilities =None,
                 sample_names=[],
                 device=None,
                 dataset_size=None,
                 indices = None,
                 ):
        '''
        sample_names : If image_selection_names is present the sample names needs to be present and have the same length.
        image_selection_names :  list of mixed strings and lists. Each sublist is a list of strings. All strings must come from the image data names.
                            when a sample is made, when a string is in image_selection_names then the sample will be taken from the
                            corresponding image data. If a list is there, then the selected name to put in the batch will be chosen at random within .

        image_selction_probabilities: List of mixed None and lists. each sublist is a list of probabilities(sum to 1).
                when selecting at random from a sublist in image_selection names, the selection will be done with the probablities given by image_selection probablities.
                If tehre is None there, then the selection will be made with equal probabilitiy

        For example :
            image_selection_names = ['seg1' , ['seg1','seg2'],['seg3','seg4']]
            image_selection_probabilities = [None, [0.2,0.8],None]
            sample_names = ['sample-seg1','sample-seg2', 'sample-seg3']

            in the sample created, then sample-seg1 will always be seg1, sample-seg2 will either be seg1 or seg2 with probablity 0.2,0.8 and sample-seg3 will
            be wither seg3 or seg4 with probability 1/2

        '''
        # Make sure arguments make sense
        if (isinstance(low_dim_data_names, str)):
            low_dim_data_names = [low_dim_data_names]

        if (isinstance(image_data_paths, str)):
            image_data_paths = [image_data_paths]
            image_data_names = [image_data_names]

        assert len(low_dim_data) == len(low_dim_data_names)
        assert len(image_data_paths) == len(image_data_names)

        if (device is None):
            self.device = torch.device('cpu')
        else:
            self.device = device

        # Load low dim data, transform them and get them to tensor
        self.low_dim_data = low_dim_data
        self.low_dim_data_names = low_dim_data_names

        self.image_data_paths = {}
        for i,name in enumerate(image_data_names):
            self.image_data_paths[name]=pickle.load(open(image_data_paths[i],'rb'))

        self.image_data_names = image_data_names

        if(len(image_selection_names) == 0):
            self.image_selection_names = self.image_data_names
            self.sample_names = self.image_data_names
        else:
            self.image_selection_names = image_selection_names
            self.sample_names = sample_names
            assert len(self.sample_names) == len(self.image_selection_names)

        if(image_selection_probabilities is None):
            self.image_selection_probabilities = [None]*len(self.image_selection_names)
        else:
            self.image_selection_probabilities = image_selection_probabilities
            assert len(image_selection_probabilities) == len(self.image_selection_names)

        if (dataset_size is None):
            self.dataset_length = len(self.image_data_paths[self.image_data_names[0]])
        else:
            self.dataset_length = dataset_size

        self.indices = indices
        if(indices is not None):
            self.dataset_length = len(indices)


    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if(self.indices is not None):
            idx = self.indices[idx]

        sample = {}

        for i, name in enumerate(self.low_dim_data_names):
            x = self.low_dim_data[i][idx]

            if(isinstance(x, np.ndarray)):
                sample[name] = torch.from_numpy(x).squeeze().float().to(self.device)
            else:
                sample[name] = torch.tensor(x).unsqueeze(0).float().to(self.device)

        for i, sample_name in enumerate(self.sample_names):

            image_name = self.image_selection_names[i]

            if(isinstance(image_name,list)):
                image_name = np.random.choice(image_name,p = self.image_selection_probabilities[i])
                #
                # #TODO: This part is hard coded and need to be aware of my naming conventions in the launching script
                # if('network' in image_name):
                #     sample['is_from_network'] = torch.tensor(1.0).unsqueeze(0).float().to(self.device)
                # else:
                #     sample['is_from_network'] = torch.tensor(0.0).unsqueeze(0).float().to(self.device)

            file_path = self.image_data_paths[image_name][idx]

            image = self.retrieve_image(file_path)

            sample[sample_name] = torch.tensor(image).to(
                self.device).float().permute(2, 0, 1)

        return sample

    def retrieve_image(self, file_path):
        image = cv2.imread(file_path, -1)
        if(image is None):
            print('FFS')

        image = image.astype(np.float32)

        if (len(image.shape) < 3):
            image = image[:, :, None]

        if (image.shape[2] == 3):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0

        return image

    def normalise_low_dim_data(self, normaliser, low_dim_data_names=None):

        if low_dim_data_names is None:
            low_dim_data_names = self.low_dim_data.keys()

        for low_dim_data_name in low_dim_data_names:
            i = self.low_dim_data_names.index(low_dim_data_name)
            self.low_dim_data[i] = normaliser.normalise(self.low_dim_data[i],  low_dim_data_name)


class DataNormaliser(object):
    def __init__(self):
        self.data_stats = {}

    def erase_data_stats(self):
        self.data_stats = {}

    def compute_and_set_data_stats(self, training_dataset, low_dim_data_names, erase_existing_data_stats=False, types = 'minmax'):
        if (erase_existing_data_stats):
            self.erase_data_stats()

        if(isinstance(types, str)):
            types = [types]*len(low_dim_data_names)

        for type_idx,  data_name in enumerate(low_dim_data_names):
            i = training_dataset.dataset.low_dim_data_names.index(data_name)
            type = types[type_idx]
            data_array = training_dataset.dataset.low_dim_data[i][training_dataset.indices]
            min = np.min(data_array, axis=0)
            max = np.max(data_array, axis=0)
            mean = np.mean(data_array, axis=0)
            std = np.std(data_array, axis=0)
            if(type == 'minmax'):
                sub = (max + min) / 2
                div = (max - min) / 2
            elif(type == '3std'):
                sub = mean
                div = 3*std
            elif(type=='std'):
                sub = mean
                div = std
            else:
                raise NotImplementedError()

            self.data_stats[data_name] = [sub, div]

    def set_data_stats(self, data_stats,
                       erase_existing_data_stats=False):

        if (erase_existing_data_stats):
            self.erase_data_stats()

        for name, value in data_stats.items():
            self.data_stats[name] = value

    def get_data_stats(self):
        return copy.deepcopy(self.data_stats)


    def normalise(self, data_array, data_name):
        '''
        Normalise te data in the data_array
        Args:
            data_array [np array] : array of data to normalise with the first dimension being the batch dimension
            data_name [str] : name of the array in the data_stats (to grab the normalisation parameters)
        '''

        sub, div = self.data_stats[data_name]

        if (isinstance(data_array, np.ndarray)):
            sub = np.stack([sub] * data_array.shape[0], axis=0)
            div = np.stack([div] * data_array.shape[0], axis=0)
            return (data_array - sub) / np.maximum((div), 1e-8)

        else:
            sub = torch.from_numpy(sub)
            div = torch.from_numpy(div)
            sub = torch.stack([sub] * data_array.shape[0], dim=0)
            div = torch.stack([div] * data_array.shape[0], dim=0)

            return (data_array - sub) / torch.maximum(div, torch.tensor(1e-8))

    def un_normalise(self, data_to_un_normalise, data_name):
        add, mul = self.data_stats[data_name]

        add = np.stack([add] * data_to_un_normalise.shape[0], axis=0)
        mul = np.stack([mul] * data_to_un_normalise.shape[0], axis=0)
        mul = np.maximum((mul), 1e-8)

        if(len(data_to_un_normalise.shape)==2 and data_to_un_normalise.shape[1]==1):
            add = add[:,None]
            mul = mul[:,None]

        if (torch.is_tensor(data_to_un_normalise)):
            add = torch.from_numpy(add).to(data_to_un_normalise.device)
            mul = torch.from_numpy(mul).to(data_to_un_normalise.device)
            return mul * data_to_un_normalise + add
        else:
            return mul * data_to_un_normalise + add


    def decode(self, data_array):
        if (isinstance(data_array, np.ndarray)):
            if(len(data_array.shape)>1):
                sins = data_array[:,0]
                cos = data_array[:,1]
                return np.arctan2(sins,cos)

            else:
                return np.arctan2(data_array[0],data_array[1])

        else:
            if (len(data_array.shape) > 1):
                sins = data_array[:, 0]
                cos = data_array[:, 1]
                return torch.atan2(sins,cos)
            else:
                return torch.atan2(data_array[0], data_array[1])








class MaskRCNNDataGrabber(Dataset):
    def __init__(self,
                 image_data_paths=[],
                 device=None,
                 dataset_size=None,
                 ):
        # Make sure arguments make sense
        assert len(image_data_paths)==2

        self.device = torch.device('cpu')

        self.image_data_names = ['rgb', 'segs']
        self.image_data_paths = {}

        for i,name in enumerate(self.image_data_names):
            self.image_data_paths[name]=pickle.load(open(image_data_paths[i],'rb'))

        if (dataset_size is None):
            self.dataset_length = len(self.image_data_paths[self.image_data_names[0]])
        else:
            self.dataset_length = dataset_size


    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        images_sample = {}

        for i, name in enumerate(self.image_data_names):
            file_path = self.image_data_paths[name][idx]

            image = self.retrieve_image(file_path)

            images_sample[name] = image

        # grab the image and seg
        image = images_sample['rgb']
        mask = images_sample['segs']

        # Grab the individual masks
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask[:,:,0] == obj_ids[:, None, None]

        # grab the bounding boxes
        num_objs = len(obj_ids)
        boxes = []

        degenerate_object_locs = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            box = np.array([xmin, ymin, xmax, ymax])

            degenerate_box = box[ 2:] <= box[ :2]
            if(degenerate_box.any()):
                degenerate_object_locs.append(i)
            else:
                boxes.append(list(box))




        masks = np.delete(masks,degenerate_object_locs,0)
        num_objs = num_objs - len(degenerate_object_locs)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return torch.tensor(image).to(
                self.device).float().permute(2, 0, 1), target

    def retrieve_image(self, file_path):
        image = cv2.imread(file_path, -1)
        if(image is None):
            print('FFS')

        image = image.astype(np.float32)

        if (len(image.shape) < 3):
            image = image[:, :, None]

        if (image.shape[2] == 3):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0

        return image
