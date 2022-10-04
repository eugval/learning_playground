from dense_correspondence_control.learning.training.propagators.propagator_base import PropagatorBase
import torch
import matplotlib.pyplot as plt
import numpy as np
import pyfastnoisesimd as fns



class DirectRegPropagator(PropagatorBase):
    def __init__(self, loss_func, optimiser, networks, segmentation_use , normaliser,
                 output= ['z_angles','scales','pixel_deltas'], shift_images = True, add_seg_noise = False):
        super().__init__(loss_func, optimiser, networks, None)
        self.segmentation_use = segmentation_use
        self.normaliser = normaliser
        self.output = output
        self.shift_images = shift_images
        self.add_seg_noise = add_seg_noise

    def _propagate_forward(self, samples):
        # Get Inputs
        bottleneck_image = samples['bottleneck_rgb']
        current_image = samples['current_rgb']
        current_seg = samples['current_seg']
        bottleneck_seg = samples['bottleneck_seg']

        if(self.add_seg_noise):
            perlin_noise_params = self.generate_perlin_noise((64,64))
            current_seg = self.adding_seg_noise(current_seg, perlin_noise_params)

        if (self.segmentation_use == 'segment'):
            bottleneck_image = bottleneck_image * bottleneck_seg
            current_image = current_image * current_seg

        elif(self.segmentation_use == 'concat'):
            bottleneck_image = torch.cat([bottleneck_image, bottleneck_seg], dim =1)
            current_image = torch.cat([current_image, current_seg], dim =1)

        output_data = []
        for type in self.output:
            output_data.append(samples[type])

        label = torch.cat(output_data, dim=1)


        # if you don't need to predict the pixel deltas, shift to center optionally
        if('pixel_deltas' not in self.output and self.shift_images ):
            current_image = self.shift_to_center(current_image, current_seg)
            bottleneck_image = self.shift_to_center(bottleneck_image, bottleneck_seg)


        # Propagate networks
        output = self.networks['model'](current_image, bottleneck_image)

        return output, label

    def train_forward_backward(self, samples):
        output, label = self._propagate_forward(samples)

        # Get Loss
        loss = self.loss_func(output, label)

        # Clear Gradients, Backrop and Optimise
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # Return Loss
        return loss.item(), {}

    def testing_forward(self, samples):
        output, label = self._propagate_forward(samples)

        # Return anything we need
        loss = self.loss_func(output, label)

        return loss.item(), output  # ,  predicted_weights , individual_losses


    def adding_seg_noise(self, segmentation_batch, perlin_noise_params):
        # Create extra segmentation mask

        number_of_artifacts = np.random.randint(0,4)
        if (number_of_artifacts == 0 and np.random.randint(3) == 0):
            return segmentation_batch

        half_seg_size = np.random.randint(2, 6)
        seg_size = segmentation_batch.shape[2]

        seg_points_x = np.random.randint(half_seg_size + 1, seg_size - half_seg_size - 1, size=(segmentation_batch.shape[0],number_of_artifacts))
        seg_points_y = np.random.randint(half_seg_size + 1, seg_size - half_seg_size - 1, size=(segmentation_batch.shape[0],number_of_artifacts))

        artifact_seg_maps = torch.zeros_like(segmentation_batch)

        for batch_index in range(segmentation_batch.shape[0]):
            for x, y in zip(seg_points_x[batch_index], seg_points_y[batch_index]):
                artifact_seg_maps[batch_index, 0, x - half_seg_size:x + half_seg_size, y - half_seg_size:y + half_seg_size] = 1.

        # combine two segmentation masks
        new_segmentation = torch.clamp(artifact_seg_maps+segmentation_batch,0.,1.)


        # add the perlin noise to each mask in the batch
        X = perlin_noise_params['X']
        Y = perlin_noise_params['Y']
        VecF0 = perlin_noise_params['VecF0']
        VecF1 = perlin_noise_params['VecF1']
        Wxy = perlin_noise_params['Wxy']

        fx = X + Wxy * VecF0
        fy = Y + Wxy * VecF1
        fx = np.where(fx < 0, 0, fx)
        fx = np.where(fx >= seg_size, seg_size-1, fx)
        fy = np.where(fy < 0, 0, fy)
        fy = np.where(fy >= seg_size, seg_size - 1, fy)
        fx = fx.astype(dtype=np.int16)
        fy = fy.astype(dtype=np.int16)

        final_segmenation = torch.zeros_like(segmentation_batch)
        for batch_index in range(new_segmentation.shape[0]):
            final_segmenation[batch_index,0,:,:] = new_segmentation[batch_index,0,fy,fx]

        zero_seg_masks = torch.sum(final_segmenation, dim =(1,2,3)) == 0
        if(zero_seg_masks.any()):
            zero_segs_indx = torch.where(zero_seg_masks)[0]
            final_segmenation[zero_segs_indx, :, :, :] = segmentation_batch[zero_segs_indx, :, :, :]
        return final_segmenation

    def shift_to_center(self, image_batch, segmentation_batch):

        batch_size, channels, h, w  = image_batch.shape

        shifted_images = []
        for i in range(batch_size):
            image = image_batch[i]
            segmentation = segmentation_batch[i]

            segmentation_median = self.get_seg_median(segmentation)
            dx = int(w //2 - segmentation_median[0].item())
            dy = int(h //2 - segmentation_median[1].item())

            image = torch.roll(image, dy, dims=1)
            image = torch.roll(image, dx, dims=2)

            shifted_images.append(image)
        return torch.stack(shifted_images,dim=0)


    def get_seg_median(self, seg):
        seg=seg.squeeze()
        x_sum = torch.sum(seg, dim=0)
        sum = torch.sum(x_sum)
        x_cumsum = torch.cumsum(x_sum, dim =0)
        x_median_pixel = torch.where(x_cumsum / sum >= 0.5)[0][0]
        y_sum = torch.sum(seg, dim=1)
        y_cumsum = torch.cumsum(y_sum, dim =0)
        y_median_pixel = torch.where(y_cumsum / sum >= 0.5)[0][0]

        current_median = torch.tensor([x_median_pixel, y_median_pixel])
        return current_median.detach()

    def generate_perlin_noise(self, bitmat_shape, seed=0):
        perlin = fns.Noise(seed=seed, numWorkers=4)

        perlin.noiseType = fns.NoiseType.SimplexFractal
        perlin.fractal.fractalType = fns.FractalType.FBM

        rndOct = np.random.choice([2, 3, 4], size=1)
        # rndOct = 8
        perlin.fractal.octaves = rndOct
        # perlin.fractal.lacunarity = 2.1
        perlin.fractal.lacunarity = np.random.uniform(2.0, 4.0)
        perlin.fractal.gain = 0.5

        perlin.perturb.perturbType = fns.PerturbType.NoPerturb

        perlin.frequency = np.random.uniform(0.005, 0.1)
        VecF0 = perlin.genAsGrid(bitmat_shape)
        perlin.frequency = np.random.uniform(0.005, 0.1)
        VecF1 = perlin.genAsGrid(bitmat_shape)

        Wxy = np.random.uniform(1, 7)
        X, Y = np.meshgrid(np.arange(bitmat_shape[1]), np.arange(bitmat_shape[0]))

        res = {
            'VecF0': VecF0,
            'VecF1': VecF1,
            'Wxy': Wxy,
            'X': X,
            'Y': Y
        }
        return res