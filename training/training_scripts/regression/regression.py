import torch
import shutil
import pickle
import numpy as np
from torch.optim import Adam
import copy
from dense_correspondence_control.utils.globals import *
from dense_correspondence_control.utils.process_utils import wait_for_pid
from dense_correspondence_control.learning.training.dataloaders import  DirectRegressionDataGrabber, DataNormaliser
from dense_correspondence_control.utils.learning_utilities import train_val_spilt
from dense_correspondence_control.learning.networks.CNN import SiameseCnnFlexible
from dense_correspondence_control.learning.training.losses import calculate_focal_loss_alpha, FocalLoss, DoubleLoss, ScaleLoss
from dense_correspondence_control.learning.training.propagators.direct_reg_propagator import DirectRegPropagator
from dense_correspondence_control.learning.training.trainer import Trainer
from dense_correspondence_control.learning.training.evaluation_functions.direct_reg_evaluation_function import test_direct_reg
from dense_correspondence_control.learning.testing.load_networks import load_network
SEED = 0



if __name__ == '__main__':
    print('PID : {}!!!'.format(os.getpid()))
    save_name = 'direct_regression/dr_scales_capped_scale_loss_noise'
    # Save training hyperparameters and training script

    if (not os.path.exists(os.path.join(MODELS_PATH, save_name))):
        os.makedirs(os.path.join(MODELS_PATH, save_name))

    shutil.copyfile(os.path.abspath(__file__), os.path.join(MODELS_PATH, save_name, 'launching_script.py'))

    wait_for =16216 # 9074 # 13231 #1150 #None ## None  # None  # Add process ID to wait for it to finish before launching  23389
    wait_for_pid(wait_for)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')

    ## Training parameters
    dataset_name = "generic_dataset_dr_64" # "cup_1k_tiny_transl  # "cup_4d_1k # cup_1k_only_rot

    training_ratio = 0.95
    learning_rate = 5e-4
    mini_batch_size = 16
    epochs = 300

    evaluation_period = 1
    model_save_period = 5

    lr_scheduler_patience = 5
    lr_scheduler_factor = 0.75
    minimum_learning_rate = 1.e-7
    weight_decay = 1e-6


    ## Create the Dataset and the train/val split
    dataset_path = os.path.join(DATASETS_PATH, dataset_name)
    bottleneck_rgb_path  =  os.path.join(dataset_path, 'bottleneck_rgb_data_paths_64.pckl')
    bottleneck_rgb_with_background_path  =  os.path.join(dataset_path, 'bottleneck_rgb_with_background_data_paths.pckl')
    bottleneck_seg_from_network_path  =  os.path.join(dataset_path, 'bottleneck_seg_from_network_data_paths.pckl')
    bottleneck_seg_path  =  os.path.join(dataset_path, 'bottleneck_object_seg_data_paths_64.pckl')
    bottleneck_norm_path  =  os.path.join(dataset_path, 'bottleneck_norm_data_paths.pckl')
    bottleneck_dist_path  =  os.path.join(dataset_path, 'bottleneck_dist_data_paths.pckl')
    bottleneck_depth_path  =  os.path.join(dataset_path, 'bottleneck_depth_data_paths.pckl')

    current_rgb_path = os.path.join(dataset_path, 'current_rgb_data_paths_64.pckl' )
    current_rgb_with_background_path = os.path.join(dataset_path, 'current_rgb_with_background_data_paths.pckl' )
    current_seg_from_network_path = os.path.join(dataset_path, 'current_seg_from_network_data_paths.pckl' )
    current_object_seg_path = os.path.join(dataset_path, 'current_object_seg_data_paths_64.pckl' )
    current_distractor_seg_path = os.path.join(dataset_path, 'current_distractor_seg_data_paths.pckl' )
    current_full_seg_path = os.path.join(dataset_path, 'current_full_seg_data_paths.pckl' )
    current_norm_path = os.path.join(dataset_path, 'current_norm_data_paths.pckl' )
    current_depth_path = os.path.join(dataset_path, 'current_depth_data_paths.pckl' )

    combined_bottleneck_rgb_path  = os.path.join(dataset_path, 'combined_bottleneck_rgb_data_paths.pckl' )
    combined_current_rgb_path  = os.path.join(dataset_path, 'combined_current_rgb_data_paths.pckl' )
    combined_current_seg_path  = os.path.join(dataset_path, 'combined_current_seg_data_paths.pckl' )

    semi_bottleneck_rgb_path  = os.path.join(dataset_path, 'semi_combined_bottleneck_rgb_data_paths.pckl' )
    semi_current_rgb_path  = os.path.join(dataset_path, 'semi_combined_current_rgb_data_clear_paths.pckl' )
    semi_current_seg_path  = os.path.join(dataset_path, 'semi_combined_current_seg_data_clear_paths.pckl' )


    z_angles_array = np.load( os.path.join(dataset_path, 'z_angles.npy'))
    z_angles_sin_cos = np.load( os.path.join(dataset_path, 'z_angles_sin_cos.npy'))
    #
    # sins = np.sin(z_angles_array)
    # cos = np.cos(z_angles_array)
    # z_angles_sin_cos = np.stack([sins, cos ],axis = 1)
    pixel_deltas_array = np.clip(np.load( os.path.join(dataset_path, 'pixel_deltas.npy')),-128,128)/2.
    scales_array = np.clip(np.load( os.path.join(dataset_path, 'scales.npy')),0.,10.)


    low_dim_data = [z_angles_sin_cos, pixel_deltas_array, scales_array]
    low_dim_data_names = ['z_angles', 'pixel_deltas', 'scales']
    low_dim_normalisation_types = [  'std', 'minmax']
    low_dim_normalisation_variables = [ 'pixel_deltas', 'scales']

    image_data_paths = [bottleneck_rgb_path, current_rgb_path, current_object_seg_path, bottleneck_seg_path] #, current_rgb_with_background_path, bottleneck_rgb_with_background_path]
    image_data_names = ["bottleneck_rgb", "current_rgb", "current_seg", "bottleneck_seg"] #, "current_bg_rgb", "bottleneck_bg_rgb"]
    #
    sample_names = [] #["bottleneck_rgb", "current_rgb", "current_seg", "bottleneck_seg"]
    image_selection_names = []# ['bottleneck_rgb', 'current_rgb', ['current_seg', 'current_network_seg'], ['bottleneck_seg', 'bottleneck_network_seg']]

    dataset = DirectRegressionDataGrabber(
                                low_dim_data = low_dim_data,
                                low_dim_data_names=low_dim_data_names,
                                image_data_paths=image_data_paths,
                                image_data_names=image_data_names,
                                image_selection_names=image_selection_names,
                                image_selection_probabilities=None,
                                sample_names=sample_names,
                                device=device,
                                  # dataset_size=None,
                                )

    training_dataset, validation_dataset =  train_val_spilt(dataset, training_ratio, SEED) #dataset,dataset #


    # create a normalise and normalise all low dim data in the dataset
    normaliser = DataNormaliser()

    # Normalise low dim data
    normaliser.compute_and_set_data_stats(training_dataset=training_dataset, low_dim_data_names=low_dim_normalisation_variables, types = low_dim_normalisation_types)
    dataset.normalise_low_dim_data(normaliser=normaliser, low_dim_data_names=low_dim_normalisation_variables)


    print('training dataset size {}'.format(len(training_dataset)))

    # Create the model
    cnn_params  = {'input_dimensions': [64,64],
                   'encoder_channels':  [3, 256, 256, 256, 256,256],   # [3, 64, 64, 64, 64, 64, 64], # [3, 32, 64, 128, 128, 256],
                   'regressor_layers': [512, 256, 256, 128, 1],    # [128, 128, 128, 4], #[512, 128, 128, 128], [16,16,16]
                    'dropout' : 0.25,
                   'double_conv': True,
                   'norm_type': 'instancenorm',
                   'final_activation': 'None',
                   'residuals': False,
                   'kernels' : None,
                    'strides' : None,
                    'activations' : 'relu',
                    'bias' : True
                    }

    # Define Networks
    model = SiameseCnnFlexible(**copy.deepcopy(cnn_params))
    model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('number of parameters: {} Million'.format(params/1000000))



    # if(load_pretrained):
    #### LOAD STATE DICT FOR CONTINUED TRAINING
    checkpoint_path = os.path.join(MODELS_PATH, 'direct_regression/dr_scales_capped_scale_loss' )
    # checkpoint_path = os.path.join(MODELS_PATH,'segmentations/seg_w_distractors_mix' )
    checkpoint_no = 245
    checkpoint_path = os.path.join(checkpoint_path, "checkpoints", "checkpoint_{}.pt".format(checkpoint_no))

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model'])
    # ####

    # loss = torch.nn.MSELoss(reduction = 'mean')
    # loss = DoubleLoss()
    normalised_1 = normaliser.normalise(np.array([1,]),'scales')
    loss = ScaleLoss(normalised_1[0])

    # Define the Optimiser
    optimiser = Adam([{'params': model.parameters()}, ], lr=learning_rate, weight_decay=weight_decay)


    testing_hyperparameters = {}
    testing_hyperparameters['model'] = {}
    testing_hyperparameters['model']['class'] = SiameseCnnFlexible
    testing_hyperparameters['model']['params'] = cnn_params

    testing_hyperparameters['normaliser'] = normaliser
    testing_hyperparameters['data_stats'] = normaliser.data_stats


    hyperparameters_save_path = os.path.join(MODELS_PATH, save_name, 'hyperparameters.pckl')
    pickle.dump(testing_hyperparameters, open(hyperparameters_save_path, 'wb'))


    # Launch Training loss_func, optimiser, networks, segmentation_use
    network_propagator = DirectRegPropagator(loss_func=loss,
                                       optimiser=optimiser,
                                       networks={'model': model},
                                        segmentation_use='segment', # 'segment', 'none' , 'concat'
                                            normaliser = normaliser,
                                            output =  [ 'scales'],  # ['z_angles'],   # ['z_angles', 'scales', 'pixel_deltas'], ['scales', 'pixel_deltas']
                                              add_seg_noise=True,
                                        )

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
        evaluation_function=test_direct_reg,
        extra_evaluation_params={
            'device':device,
            'use_sin_cos':True
        },
        save_name=save_name,
        # training_evaluation_function=don_in_training_evaluation
    )

    trainer.train(scheduler)
    # trainer.evaluate()
    print('DONE')