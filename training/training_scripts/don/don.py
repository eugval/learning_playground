import torch
import shutil
import pickle
import numpy as np
from torch.optim import Adam
import copy
from dense_correspondence_control.utils.globals import *
from dense_correspondence_control.utils.process_utils import wait_for_pid
from dense_correspondence_control.learning.training.dataloaders import  CorrespondenceDataGrabber
from dense_correspondence_control.utils.learning_utilities import train_val_spilt
from dense_correspondence_control.learning.networks.Unet import Unet
from dense_correspondence_control.learning.training.losses import ContrastiveDonLoss
from dense_correspondence_control.learning.training.propagators.don_propagator import DonPropagator
from dense_correspondence_control.learning.training.trainer import Trainer
from dense_correspondence_control.learning.training.evaluation_functions.don_evaluation_function import test_don, don_in_training_evaluation

SEED = 0

if __name__ == '__main__':
    wait_for = None  # Add process ID to wait for it to finish before launching
    wait_for_pid(wait_for)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    ## Training parameters
    dataset_name = "don_many_obj_randomised" # "cup_1k_tiny_transl  # "cup_4d_1k # cup_1k_only_rot

    training_ratio = 0.95
    learning_rate = 1e-3
    mini_batch_size = 1
    epochs = 400

    evaluation_period = 1
    model_save_period = 5

    lr_scheduler_patience = 5
    lr_scheduler_factor = 0.75
    minimum_learning_rate = 1.e-7

    save_name = 'DoN/many_objects'
    #Note: no norm here
    # pretrained_path = None
    # pretrained_checkpoint =

    ## Create the Dataset and the train/val split
    dataset_path = os.path.join(DATASETS_PATH, dataset_name)
    image_path =  os.path.join(dataset_path, 'images')
    tensor_path = os.path.join(dataset_path, 'tensors' )

    from_image_names = os.path.join(tensor_path,'full_f_from')
    to_image_names = os.path.join(tensor_path,'full_f_to')

    from_x = os.path.join(tensor_path, 'from_correct_xs')
    from_y = os.path.join(tensor_path, 'from_correct_ys')
    to_x =  os.path.join(tensor_path, 'to_correct_xs')
    to_y =  os.path.join(tensor_path, 'to_correct_ys')

    from_w_x = os.path.join(tensor_path, 'from_wrong_xs')
    from_w_y = os.path.join(tensor_path, 'from_wrong_ys')
    to_w_x = os.path.join(tensor_path, 'to_wrong_xs')
    to_w_y = os.path.join(tensor_path, 'to_wrong_ys')

    from_w_o_x = os.path.join(tensor_path, 'from_wrong_object_xs')
    from_w_o_y = os.path.join(tensor_path, 'from_wrong_object_ys')
    to_w_o_x = os.path.join(tensor_path, 'to_wrong_object_xs')
    to_w_o_y = os.path.join(tensor_path, 'to_wrong_object_ys')


    low_dim_data_paths = [from_x, from_y, to_x, to_y, from_w_x, from_w_y, to_w_x,to_w_y,from_w_o_x,from_w_o_y,to_w_o_x, to_w_o_y]
    low_dim_data_names = ['from_x', 'from_y', 'to_x', 'to_y', 'from_w_x', 'from_w_y', 'to_w_x','to_w_y','from_w_o_x','from_w_o_y','to_w_o_x', 'to_w_o_y']

    image_data_folders = [image_path, image_path]
    image_data_names = ["rgb_from", "rgb_to"]
    image_filename_paths = [from_image_names, to_image_names]
    image_data_types = ['.png', '.png']


    dataset = CorrespondenceDataGrabber(
                                low_dim_data_paths =low_dim_data_paths,
                                low_dim_data_names = low_dim_data_names,
                                image_filename_paths = image_filename_paths,
                                image_data_folders = image_data_folders,
                                image_data_names =image_data_names,
                                image_data_types = image_data_types,
                                device = device,
                                dataset_size = None,
                                )

    training_dataset, validation_dataset =  train_val_spilt(dataset, training_ratio, SEED) #dataset,dataset #

    # Create the model
    Unet_params  = {'encoder_channels' :  [3, 32, 64, 128, 256, 256, 512],
                    'decoder_channels':   [512, 256, 256, 128, 64, 32],
                    'parametric_decoder': False,
                    'activations': 'relu',
                    'bias': True,
                  'norm_type':'instancenorm',
                    'dropout':0.,
                     'residuals':False,
                    'double_conv':False,
                    'use_1x1_conv':True,
                    'concat_rgb': True,
                  'final_activation':'none',
                     'final_encoder_activation':'relu',
                  'double_convs_second_activation':None,
                     'num_output_channels':3,
                    'norm_layer_arguments':{}, #{'track_running_stats': False,}, #
                    'post_upsampling_convs' : [(128,3,1), (128,3,1), (128,3,1)]}

    # Define Networks
    model = Unet(**copy.deepcopy(Unet_params))
    model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('number of parameters: {} Million'.format(params/1000000))



    # if(load_pretrained):
    #
    #
    #


    # Define Loss
    loss = ContrastiveDonLoss(distance_metric='double',
                              margin_bg= 15.,
                              margin_obj= 15.,
                              hard_negative_scaling=True)

    # Define the Optimiser
    optimiser = Adam([{'params': model.parameters()}, ], lr=learning_rate)

    # Save training hyperparameters and training script
    if (not os.path.exists(os.path.join(MODELS_PATH, save_name))):
        os.makedirs(os.path.join(MODELS_PATH, save_name))

    shutil.copyfile(os.path.abspath(__file__), os.path.join(MODELS_PATH, save_name, 'launching_script.py'))

    testing_hyperparameters = {}
    testing_hyperparameters['model'] = {}
    testing_hyperparameters['model']['class'] = Unet
    testing_hyperparameters['model']['params'] = Unet_params


    hyperparameters_save_path = os.path.join(MODELS_PATH, save_name, 'hyperparameters.pckl')
    pickle.dump(testing_hyperparameters, open(hyperparameters_save_path, 'wb'))


    # Launch Training
    network_propagator = DonPropagator(loss_func=loss,
                                       optimiser=optimiser,
                                       networks={'model': model})

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min',
                                                           factor=lr_scheduler_factor,
                                                           patience=lr_scheduler_patience,
                                                           min_lr=minimum_learning_rate,
                                                           verbose=True)

    trainer = Trainer(network_propagator=network_propagator,
                      training_dataset=training_dataset,
                      validation_dataset=validation_dataset,
                      mini_batch_size=mini_batch_size,
                      epochs=epochs,
                      evaluation_period=evaluation_period,
                      model_save_period=model_save_period,
                      evaluation_function=test_don,
                      extra_evaluation_params={'image_save_path': os.path.join(MODELS_PATH,save_name, 'image_examples')},
                      save_name=save_name,
                      #training_evaluation_function=don_in_training_evaluation
                      )

    trainer.train(scheduler)
    print('DONE')
