import torch
import shutil
import pickle
import numpy as np
from torch.optim import Adam
import copy
from dense_correspondence_control.utils.globals import *
from dense_correspondence_control.utils.process_utils import wait_for_pid
from dense_correspondence_control.learning.training.dataloaders import  NewSegmentationDataGrabber
from dense_correspondence_control.utils.learning_utilities import train_val_spilt
from dense_correspondence_control.learning.networks.Unet import Unet
from dense_correspondence_control.learning.training.losses import calculate_focal_loss_alpha, FocalLoss
from dense_correspondence_control.learning.training.propagators.seg_propagator import SimpleSegmentationPropagator
from dense_correspondence_control.learning.training.trainer import Trainer
from dense_correspondence_control.learning.training.evaluation_functions.simple_seg_evaluation_function import test_seg
SEED = 0

if __name__ == '__main__':
    print('PID : {}!!!'.format(os.getpid()))
    wait_for =  16779# Add process ID to wait for it to finish before launching
    wait_for_pid(wait_for)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu') #if torch.cuda.is_available() else torch.device('cpu')

    ## Training parameters
    dataset_name = "cup_only_segmentation" # "cup_1k_tiny_transl  # "cup_4d_1k # cup_1k_only_rot

    training_ratio = 0.95
    learning_rate = 1e-3
    mini_batch_size = 32
    epochs = 150

    evaluation_period = 1
    model_save_period = 5

    lr_scheduler_patience = 5
    lr_scheduler_factor = 0.75
    minimum_learning_rate = 1.e-7

    save_name = 'segmentations/simple_segmentation_2m_normalised_no_l2'
    #Note: no norm here
    # pretrained_path = None
    # pretrained_checkpoint =

    ## Create the Dataset and the train/val split
    dataset_path = os.path.join(DATASETS_PATH, dataset_name)
    bottleneck_rgb_path  =  os.path.join(dataset_path, 'bottleneck_rgb_data_paths.pckl')
    bottleneck_seg_path  =  os.path.join(dataset_path, 'bottleneck_seg_data_paths.pckl')

    current_rgb_path  = os.path.join(dataset_path, 'current_rgb_data_paths.pckl' )
    current_seg_path  = os.path.join(dataset_path, 'current_seg_data_paths.pckl')

    low_dim_data_paths = []
    low_dim_data_names = []

    image_data_folders = [ current_rgb_path, current_seg_path]
    image_data_names = [ "current_rgb", "current_seg"]


    dataset = NewSegmentationDataGrabber(
                                low_dim_data_paths = low_dim_data_paths,
                                low_dim_data_names = low_dim_data_names,
                                image_data_paths = image_data_folders,
                                image_data_names = image_data_names,
                                device=device,
                              # dataset_size=150,
                                )

    training_dataset, validation_dataset =  train_val_spilt(dataset, training_ratio, SEED) #dataset,dataset #

    # Create the model
    Unet_params  = {'encoder_channels' :  [3, 16, 32, 64, 128, 128, 256],
                    'decoder_channels':   [256, 128, 128, 64, 32, 32],
                    'parametric_decoder': False,
                    'activations': 'relu',
                    'bias': True,
                    'norm_type': 'instancenorm',
                    'dropout': 0.25,
                    'residuals': False,
                    'double_conv': False,
                    'use_1x1_conv': True,
                    'concat_rgb': True,
                    'final_activation': 'sigmoid',
                    'final_encoder_activation': 'relu',
                    'double_convs_second_activation': None,
                    'num_output_channels': 1,
                    'norm_layer_arguments': {},  # {'track_running_stats': False,}, #
                    'post_upsampling_convs': [(64, 3, 1), (64, 3, 1), (64, 3, 1)],
                    'return_features_pre_final_activation': True,
                    'post_decoder_norm': True,
                    }

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
    mask_files = pickle.load(open(current_seg_path,'rb'))
    alpha = calculate_focal_loss_alpha(mask_files=mask_files)
    # Define Loss
    loss = FocalLoss(
        alpha=alpha,
        gamma=2,
        epsilon = 1e-8,
        logits = True,
        device = device,
    )

    # Define the Optimiser
    optimiser = Adam([{'params': model.parameters()}, ], lr=learning_rate) #, weight_decay= 1e-6

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
    network_propagator = SimpleSegmentationPropagator(loss_func=loss,
                                       optimiser=optimiser,
                                       networks={'model': model},
                                        normalised= True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min',
                                                           factor=lr_scheduler_factor,
                                                           patience=lr_scheduler_patience,
                                                           min_lr=minimum_learning_rate,
                                                           verbose=True)

    trainer = Trainer(
        network_propagator=network_propagator,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        mini_batch_size=mini_batch_size,
        epochs=epochs,
        evaluation_period=evaluation_period,
        model_save_period=model_save_period,
        evaluation_function=test_seg,
        extra_evaluation_params={
            'sim_save_path': os.path.join(MODELS_PATH, save_name, 'sim_examples'),
            'easy_real_save_path': os.path.join(MODELS_PATH, save_name, 'easy_real_examples'),
            'hard_real_save_path': os.path.join(MODELS_PATH, save_name, 'hard_real_examples'),
            'easy_test_set_path': os.path.join(DATA_PATH, 'test_data/easy_seg_test_set'),
            'hard_test_set_path': os.path.join(DATA_PATH, 'test_data/hard_seg_test_set'),
        },
        save_name=save_name,
        # training_evaluation_function=don_in_training_evaluation
    )

    trainer.train(scheduler)
    print('DONE')
