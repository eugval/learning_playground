import cv2
import numpy as np
import glob
import time
import os
import torch
import matplotlib.pyplot as plt
import gc
import sys
import time

def load_correspondance(file,dim_im , ratio_dim):
    im = cv2.imread(file, -1)
#    im = im[162:362, 162:362]
    im = cv2.resize(im, (dim_im,dim_im))//ratio_dim
    return im

def load_rgb(file, dim_im):
    im = cv2.imread(file, -1)
 #   im = im[162:362, 162:362]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (dim_im,dim_im))
    return im
def load_semantic(file, dim_im):
    im = cv2.imread(file, -1)
 #   im = im[162:362, 162:362]
    im = cv2.resize(im, (dim_im,dim_im))
    return im
def load_segmentation(file, dim_im, index_to_consider = None):
    im = cv2.imread(file, -1)
 #   im = im[162:362, 162:362]
    if(index_to_consider is not None):
        index_to_consider = int(index_to_consider)
        im = (im == index_to_consider).astype('float')

    if(np.isclose(im,0.).all()):
        print("Alert !! Object not detected in image, double check the segmentation index is correct. Image: {}".format(file))

    im = np.round(cv2.resize(im, (dim_im,dim_im)))
    return im


### TODO: Deprecated?  ###
# def store_rgb(org_fold, dim_im, write_folder="data/", n_folders=100):
#     rgb_froms = np.empty((45 * n_folders, dim_im, dim_im, 3), dtype=np.uint8)
#     rgb_tos = np.empty((45 * n_folders, dim_im, dim_im, 3), dtype=np.uint8)
#
#     k = 0
#
#     for subfolder in range(n_folders):  # should be 100
#         subfolder = str(subfolder)
#         if len(subfolder) == 1:
#             sub_name = "00000{}".format(subfolder)
#         elif len(subfolder) == 2:
#             sub_name = "0000{}".format(subfolder)
#         elif len(subfolder) == 3:
#             sub_name = "000{}".format(subfolder)
#
#         print("f", org_fold)
#         print("subname")
#         folder = org_fold + sub_name
#         print("folder", folder)
#         files = glob.glob(folder + "/correspondance/*")
#         print(len(files))
#         for f in files:
#             parts = f.split("_")
#             from_file = parts[-3]
#             to_file = parts[-1].split(".")[0]
#
#             rgb_from = load_rgb(folder + "/rgb/" + from_file + ".jpg")
#             rgb_to = load_rgb(folder + "/rgb/" + to_file + ".jpg")
#
#             cv2.imwrite(write_folder + "{}_{}.jpg".format(subfolder, from_file), rgb_from)
#
#             k += 1


### TODO: Deprecated? ####
# def get_random_mappings_parallel(segmentation, correspondance_sem, correspondance):
#     start = time.time()
#  #   to_ys, to_xs, from_ys, from_xs = [], [], [], []
#     valid_ys, valid_xs = tnp.nonzero(((segmentation == 1)&(correspondance_sem == 3))*1 )
#     if len(valid_ys) == 0:
#         return 0,0,0,0, False
#     valids = tnp.nonzero((correspondance_sem == 3)*1.)
#     tries = 0
#     tries += 1
#     rand_ind = tnp.random.randint(0, len(valid_ys))
#     valid_y, valid_x = valid_ys[rand_ind], valid_xs[rand_ind]
#     if tries > 100: #this  should be useless now but I'll keep it to see if it ever fails
#         print("Failed")
#         return 0,0,0,0, False
#     corr_x, corr_y, _ = correspondance[valid_y, valid_x]
#   #  from_ys.append(valid_y), from_xs.append(valid_x), to_ys.append(int(corr_y)), to_xs.append(int(corr_x))
#     from_ys, from_xs, to_ys, to_xs = valid_y, valid_x, corr_y, corr_x
#   #  print("correct mapping", time.time() - start)
#     return from_ys, from_xs, to_ys, to_xs, True
#
#


def create_full_data(org_fold, write_folder, dim_im, ratio_dim,  n_folders=100, starting_folder = 0, write_folder_seg = None, segmentation_index =1 ):
    print('HERE')
    semantic_froms = np.empty((45 * n_folders, dim_im, dim_im), dtype=np.uint8)
    correspondances = np.empty((45 * n_folders, dim_im, dim_im, 3), dtype=np.float32)
    segmentation_froms = np.empty((45 * n_folders, dim_im, dim_im), dtype=np.uint8)
    files_corr_from_org = []
    files_corr_to_org = []
    k = 0

    for subfolder in range(n_folders):  # should be 100
        subfolder = starting_folder + subfolder
        subfolder = str(subfolder)
        if len(subfolder) == 1:
            sub_name = "00000{}".format(subfolder)
        elif len(subfolder) == 2:
            sub_name = "0000{}".format(subfolder)
        elif len(subfolder) == 3:
            sub_name = "000{}".format(subfolder)
        elif len(subfolder) == 4:
            sub_name = "00{}".format(subfolder)
        elif len(subfolder) == 5:
            sub_name = "0{}".format(subfolder)
        elif len(subfolder) == 6:
            sub_name = "{}".format(subfolder)



        folder = org_fold + sub_name
        files = glob.glob(folder + "/correspondance/*")

        for f in files:
            parts = f.split("_")
            from_file = parts[-3]
            to_file = parts[-1].split(".")[0]
            corr = load_correspondance(f, dim_im , ratio_dim)
            rgb_from = load_rgb(folder + "/rgb/" + from_file + ".png", dim_im) # TODO: CHANGE THAT BACK TO JPEG
            rgb_to = load_rgb(folder + "/rgb/" + to_file + ".png", dim_im)


            if(write_folder_seg is not None):
                seg_from = load_segmentation(folder+'/segmentation/'+from_file+'.png', dim_im, index_to_consider=segmentation_index)
                seg_to = load_segmentation(folder+'/segmentation/'+to_file+'.png', dim_im, index_to_consider=segmentation_index)


            semantic_from = load_semantic(
                folder + "/correspondance_semantic_map/from_{}_to_{}.png".format(from_file, to_file), dim_im)
            segmentation_from = load_segmentation(folder + "/segmentation/" + from_file + ".png", dim_im, index_to_consider= segmentation_index)
            mapping = load_correspondance(folder + "/correspondance/from_{}_to_{}.png".format(from_file, to_file), dim_im, ratio_dim)
            files_corr_from_org.append("{}_{}".format(sub_name, from_file))
            files_corr_to_org.append("{}_{}".format(sub_name, to_file))
            cv2.imwrite(write_folder + "{}_{}.png".format(sub_name, from_file), cv2.cvtColor(rgb_from, cv2.COLOR_RGB2BGR))
            cv2.imwrite(write_folder + "{}_{}.png".format(sub_name, to_file), cv2.cvtColor(rgb_to, cv2.COLOR_RGB2BGR))

            if(write_folder_seg is not None):
                assert (np.logical_or(np.unique(seg_to)==0., np.unique(seg_to)==1.)).all()
                assert (np.logical_or(np.unique(seg_from)==0. , np.unique(seg_from)==1.)).all()
                # if(sub_name ==  '000019' and to_file == '000002'):
                #     print('fuck you')
                #     print(np.unique(seg_to))
                cv2.imwrite(write_folder_seg + "{}_{}.png".format(sub_name, from_file), seg_from.astype('uint8'))
                cv2.imwrite(write_folder_seg + "{}_{}.png".format(sub_name, to_file), seg_to.astype('uint8'))

                # if (sub_name == '000019' and to_file == '000002'):
                #     print('fuck yo u aginag')
                #     a = cv2.imread(write_folder_seg + "{}_{}.png".format(sub_name, to_file), -1).astype(np.float32)
                #     print(np.unique(a))

            semantic_froms[k, :] = semantic_from
            segmentation_froms[k, :] = segmentation_from
            correspondances[k, :] = mapping
            k += 1
    return correspondances[:k], semantic_froms[:k], segmentation_froms[:k], \
           files_corr_from_org, files_corr_to_org


def get_wrong_object_mappings(segmentation, correspondance_sem, correspondance, samples=10):
    to_ys, to_xs, from_ys, from_xs = np.zeros(samples, dtype=np.int32), np.zeros(samples, dtype=np.int32), np.zeros(
        samples, dtype=np.int32), np.zeros(samples, dtype=np.int32)
    random_placement = np.arange(0, samples)
    np.random.shuffle(random_placement)
    valid_ys, valid_xs = np.nonzero((segmentation == 1) * 1.)

    #  valid_ys, valid_xs = np.nonzero(((segmentation == 1)&(correspondance_sem == 3))*1 )
    if len(valid_ys) == 0:
        return 0, 0, 0, 0, False
    # valids = np.nonzero((correspondance_sem == 3)*1.)

    for i in range(samples):
        tries = 0

        valid = 1
        distance = 0
        while distance < 2:
            tries += 1
            rand_ind = np.random.randint(0, len(valid_ys))
            rand_ind_2 = np.random.randint(0, len(valid_ys))
            valid_y, valid_x = valid_ys[rand_ind], valid_xs[rand_ind]
            valid_y_2, valid_x_2 = valid_ys[rand_ind_2], valid_xs[rand_ind_2]
            distance = np.sqrt((valid_y - valid_y_2) ** 2) + np.sqrt((valid_x - valid_x_2) ** 2)

            if tries > 1000:  # this  should be useless now but I'll keep it to see if it ever fails
                print("Failed")
                return 0, 0, 0, 0, False
        from_ys[i] = int(valid_y)
        from_xs[i] = int(valid_x)
        to_ys[i] = int(valid_y_2)
        to_xs[i] = int(valid_x_2)
    return from_ys, from_xs, to_ys, to_xs, True


def get_random_mappings(segmentation, correspondance_sem, correspondance, samples=10):
    #   to_ys, to_xs, from_ys, from_xs = [], [], [], []
    to_ys, to_xs, from_ys, from_xs = np.empty((samples), dtype=np.int32), np.empty((samples), dtype=np.int32), np.empty(
        (samples), dtype=np.int32), np.empty((samples), dtype=np.int32)
    valid_ys, valid_xs = np.nonzero(((segmentation == 1) & (correspondance_sem == 3)) * 1)
    if len(valid_ys) == 0:
        return 0, 0, 0, 0, False
    valids = np.nonzero((correspondance_sem == 3) * 1.)
    for i in range(samples):
        valid = 1
        tries = 0
        while not valid == 3:
            tries += 1
            rand_ind = np.random.randint(0, len(valid_ys))
            valid_y, valid_x = valid_ys[rand_ind], valid_xs[rand_ind]
            valid = correspondance_sem[valid_y, valid_x]
            if tries > 100:  # this  should be useless now but I'll keep it to see if it ever fails
                print("Failed")
                return 0, 0, 0, 0, False
        corr_x, corr_y, _ = correspondance[valid_y, valid_x]
        #  from_ys.append(valid_y), from_xs.append(valid_x), to_ys.append(int(corr_y)), to_xs.append(int(corr_x))
        from_ys[i], from_xs[i], to_ys[i], to_xs[i] = valid_y, valid_x, corr_y, corr_x

    return from_ys, from_xs, to_ys, to_xs, True


def get_wrong_random_mappings(segmentation, correspondance_sem, correspondance, dim_im, samples=10):
    # to_ys, to_xs, from_ys, from_xs = [], [], [], []
    to_ys, to_xs, from_ys, from_xs = np.empty((samples), dtype=np.int32), np.empty((samples), dtype=np.int32), np.empty(
        (samples), dtype=np.int32), np.empty((samples), dtype=np.int32)
    valid_ys, valid_xs = np.nonzero((segmentation == 1) * 1.)
    if len(valid_ys) == 0:
        return 0, 0, 0, 0, False
    valids = np.nonzero((correspondance_sem == 3) * 1.)
    bg_ys, bg_xs = np.nonzero((segmentation==0) * 1.)
    for i in range(samples):

        if i < samples // 2:
            valid = 1

            rand_ind = np.random.randint(0, len(valid_ys))
            valid_y, valid_x = valid_ys[rand_ind], valid_xs[rand_ind]
            rand_bg_ind = np.random.randint(0, len(bg_ys))
            corr_x, corr_y, _ = bg_ys[rand_bg_ind], bg_xs[rand_bg_ind], 0
            # from_ys.append(valid_y), from_xs.append(valid_x)
            # to_ys.append(int(corr_y))
            # to_xs.append(int(corr_x))
            from_ys[i], from_xs[i], to_ys[i], to_xs[i] = valid_y, valid_x, corr_y, corr_x
        else:
            # Take close to the object
            is_bg = False
            while not is_bg:
                rand_ind = np.random.randint(0, len(valid_ys))
                valid_y, valid_x = valid_ys[rand_ind], valid_xs[rand_ind]
                bg_y, bg_x = valid_y + np.random.randint(-20, 20), valid_x + np.random.randint(-20, 20)
                bg_y, bg_x = np.clip(bg_y, a_min=0, a_max=dim_im - 1), np.clip(bg_x, a_min=0, a_max=dim_im - 1)
                is_bg = segmentation[bg_y, bg_x] == 0
            from_ys[i], from_xs[i], to_ys[i], to_xs[i] = valid_y, valid_x, bg_y, bg_x
    # print("W BG mapping", time.time() - start)
    return from_ys, from_xs, to_ys, to_xs, True


def data_in_tensors(x):
    try:
        fr, to, samples, samples_wo, samples_w, correspondances, semantic_froms, segmentation_froms, files_corr_from, files_corr_to, dim_im = x
        size = to - fr
        from_correct_xs, from_correct_ys, to_correct_xs, to_correct_ys, \
        from_wrong_object_xs, from_wrong_object_ys, to_wrong_object_xs, to_wrong_object_ys, \
        from_wrong_xs, from_wrong_ys, to_wrong_xs, to_wrong_ys = \
            np.empty((size, samples)), np.empty((size, samples)), np.empty((size, samples)), \
            np.empty((size, samples)), np.empty((size, samples_wo)), np.empty((size, samples_wo)), \
            np.empty((size, samples_wo)), np.empty((size, samples_wo)), np.empty((size, samples_w)), \
            np.empty((size, samples_w)), np.empty((size, samples_w)), np.empty((size, samples_w))
        files_corr_from_new, files_corr_to_new = [], []

        i = 0
        for j in range(fr, to):
            corr, semantic_from, segmentation_from = correspondances[j], semantic_froms[j], segmentation_froms[j]

            from_ys1, from_xs1, to_ys1, to_xs1, success1 = get_random_mappings(segmentation_from, semantic_from, corr,
                                                                               samples=samples)
            from_ys2, from_xs2, to_ys2, to_xs2, success2 = get_wrong_object_mappings(segmentation_from, semantic_from,
                                                                                     corr, samples=samples_wo)
            from_ys3, from_xs3, to_ys3, to_xs3, success3 = get_wrong_random_mappings(segmentation_from, semantic_from,
                                                                                     corr, dim_im=dim_im, samples=samples_w)

            if success1 and success2 and success3:
                from_correct_xs[i, :], from_correct_ys[i, :], to_correct_xs[i, :], to_correct_ys[i, :] = \
                    from_xs1, from_ys1, to_xs1, to_ys1

                # test_correspondance(rgb_from, rgb_to, from_xs, from_ys, to_xs, to_ys)

                from_wrong_object_xs[i, :], from_wrong_object_ys[i, :], to_wrong_object_xs[i, :], to_wrong_object_ys[i,
                                                                                                  :] = \
                    from_xs2, from_ys2, to_xs2, to_ys2

                from_wrong_xs[i, :], from_wrong_ys[i, :], to_wrong_xs[i, :], to_wrong_ys[i, :] = \
                    from_xs3, from_ys3, to_xs3, to_ys3

                files_corr_from_new.append(files_corr_from[j])
                files_corr_to_new.append(files_corr_to[j])

                #   rgb_tos_tensor[i,:] = rgb_to
                #   print(rgb_tos_tensor[i,:])
                #   rgb_froms_tensor[i,:] = rgb_from

                i += 1
    except Exception as e:
        print(e)

    return from_correct_xs[:i], from_correct_ys[:i], to_correct_xs[:i], to_correct_ys[:i], \
           from_wrong_object_xs[:i], from_wrong_object_ys[:i], to_wrong_object_xs[:i], to_wrong_object_ys[:i], \
           from_wrong_xs[:i], from_wrong_ys[:i], to_wrong_xs[:i], to_wrong_ys[:i], files_corr_from_new[
                                                                                   :i], files_corr_to_new[:i]


def setup_tensors(r, multiprocessing = True):
    if(multiprocessing):
        from_correct_xs, from_correct_ys, to_correct_xs, to_correct_ys, \
        from_wrong_object_xs, from_wrong_object_ys, to_wrong_object_xs, to_wrong_object_ys, \
        from_wrong_xs, from_wrong_ys, to_wrong_xs, to_wrong_ys, files_corr_from, files_corr_to = [
            np.concatenate([r[i][j] for i in range(len(r))]) for j in range(14)]
    else:
        from_correct_xs, from_correct_ys, to_correct_xs, to_correct_ys, \
        from_wrong_object_xs, from_wrong_object_ys, to_wrong_object_xs, to_wrong_object_ys, \
        from_wrong_xs, from_wrong_ys, to_wrong_xs, to_wrong_ys, files_corr_from, files_corr_to = [r[j] for j in range(14)]

    return from_correct_xs.astype(int), from_correct_ys.astype(int), to_correct_xs.astype(int), to_correct_ys.astype(
        int), \
           from_wrong_object_xs.astype(int), from_wrong_object_ys.astype(int), to_wrong_object_xs.astype(
        int), to_wrong_object_ys.astype(int), \
           from_wrong_xs.astype(int), from_wrong_ys.astype(int), to_wrong_xs.astype(int), to_wrong_ys.astype(
        int), files_corr_from, files_corr_to


def save_data(x, fr=0, save_in_one_folder=True):
    tensor, bs, name = x
    steps = len(tensor) // bs
    for i in range(fr, steps + fr):
        if (save_in_one_folder):
            torch.save(tensor[i * bs:(i + 1) * bs], "{}_{}.pt".format(name, i))
        else:
            folder = "{}/".format(name)
            file = os.path.join(folder, '{}.pt'.format(i))

            if (not os.path.exists(folder)):
                os.makedirs(folder)
            batch_index = i-fr
            torch.save(tensor[batch_index * bs:(batch_index + 1) * bs], file)

    return steps + fr




if __name__ == '__main__':
    from multiprocessing import Pool
    import time
    import os
    from contextlib import closing

    gc.enable()


    offset_pixels_x, offset_pixels_y = 162, 162
    dim_im = 128
    ratio_dim = 128 / dim_im

    # Where to fetch the data

    folder = "/home/eugene/Projects/BlenderProc/examples/data/datasets/multi_object_randomised_textured/"
    print(os.path.exists(folder))

    # Where to save the data
    write_folder_base = "/home/eugene/Projects/dense_correspondence_control/dense_correspondence_control/learning/data/datasets/don_many_obj_randomised_texured/"
    write_folder_images = write_folder_base + 'images/'  # For original data organisation, use write_folder_images = write_folder_base
    write_folder_seg = write_folder_base + 'seg/'

    if not (os.path.exists(write_folder_images)):
        os.makedirs(write_folder_images)

    if not (os.path.exists(write_folder_seg)):
        os.makedirs(write_folder_seg)


    # Do a large dataset in chunks
    total_amount_of_folders = 900
    n_folders_per_time = 100
    segmentation_index = 7

    samples = 2000
    samples_wo = 5000
    samples_w = 1500

    number_of_runs = int(total_amount_of_folders / n_folders_per_time)


    total_files_from = np.array([])
    total_files_to = np.array([])

    for run in range(number_of_runs):
        print('Run no {}'.format(run))

        start = time.time()

        initial_folder_index = run * n_folders_per_time
        correspondances, semantic_froms, segmentation_froms, files_corr_from, files_corr_to = create_full_data(folder,
                                                                                                               write_folder_images,
                                                                                                               dim_im,
                                                                                                               ratio_dim,
                                                                                                               write_folder_seg=write_folder_seg,
                                                                                                               n_folders=n_folders_per_time,
                                                                                                               starting_folder=initial_folder_index,
                                                                                                               segmentation_index=segmentation_index)


        print('Established data, took {} minutes, making tensors...'.format((time.time()-start)/60))
        print('number_of datapoints {}'.format(len(correspondances)))

        start = time.time()
        # from_data = 0 * len(correspondances)
        # to_data = len(correspondances)
        # procs = 5
        # step = (to_data - from_data) // procs
        #
        # with closing(Pool(procs,maxtasksperchild=1)) as p:
        #     r = p.map(data_in_tensors, [
        #         [from_data + step * i, from_data + step * (i + 1), samples, samples_wo, samples_w, correspondances,
        #          semantic_froms, segmentation_froms, files_corr_from, files_corr_to, dim_im] for i in range(procs)])

        from_data = 0 * len(correspondances)
        to_data = len(correspondances)

        r = data_in_tensors(  [from_data, to_data, samples, samples_wo, samples_w, correspondances,
                 semantic_froms, segmentation_froms, files_corr_from, files_corr_to, dim_im])



        from_correct_xs, from_correct_ys, to_correct_xs, to_correct_ys, \
        from_wrong_object_xs, from_wrong_object_ys, to_wrong_object_xs, to_wrong_object_ys, \
        from_wrong_xs, from_wrong_ys, to_wrong_xs, to_wrong_ys, files_corr_from, files_corr_to = setup_tensors(r, multiprocessing = False)

        print('Tensors Ready,took {} minutes saving...'.format((time.time()-start)/60))

        save_in_one_folder = False
        initial_index = len(total_files_from)
        batch_size = 1

        save_data([from_correct_xs, batch_size, write_folder_base + "/tensors/from_correct_xs"], fr=initial_index,
                  save_in_one_folder=save_in_one_folder)
        save_data([from_correct_ys, batch_size, write_folder_base + "/tensors/from_correct_ys"], fr=initial_index,
                  save_in_one_folder=save_in_one_folder)
        save_data([to_correct_xs, batch_size, write_folder_base + "/tensors/to_correct_xs"], fr=initial_index,
                  save_in_one_folder=save_in_one_folder)
        save_data([to_correct_ys, batch_size, write_folder_base + "/tensors/to_correct_ys"], fr=initial_index,
                  save_in_one_folder=save_in_one_folder)
        save_data([from_wrong_object_xs, batch_size, write_folder_base + "/tensors/from_wrong_object_xs"],
                  fr=initial_index, save_in_one_folder=save_in_one_folder)
        save_data([from_wrong_object_ys, batch_size, write_folder_base + "/tensors/from_wrong_object_ys"],
                  fr=initial_index, save_in_one_folder=save_in_one_folder)
        save_data([to_wrong_object_xs, batch_size, write_folder_base + "/tensors/to_wrong_object_xs"], fr=initial_index,
                  save_in_one_folder=save_in_one_folder)
        save_data([to_wrong_object_ys, batch_size, write_folder_base + "/tensors/to_wrong_object_ys"], fr=initial_index,
                  save_in_one_folder=save_in_one_folder)
        save_data([from_wrong_xs, batch_size, write_folder_base + "/tensors/from_wrong_xs"], fr=initial_index,
                  save_in_one_folder=save_in_one_folder)
        save_data([from_wrong_ys, batch_size, write_folder_base + "/tensors/from_wrong_ys"], fr=initial_index,
                  save_in_one_folder=save_in_one_folder)
        save_data([to_wrong_xs, batch_size, write_folder_base + "/tensors/to_wrong_xs"], fr=initial_index,
                  save_in_one_folder=save_in_one_folder)
        save_data([to_wrong_ys, batch_size, write_folder_base + "/tensors/to_wrong_ys"], fr=initial_index,
                  save_in_one_folder=save_in_one_folder)
        save_data([files_corr_from, batch_size, write_folder_base + "/tensors/files_corr_from"], fr=initial_index,
                  save_in_one_folder=save_in_one_folder)
        save_data([files_corr_to, batch_size, write_folder_base + "/tensors/files_corr_to"], fr=initial_index,
                  save_in_one_folder=save_in_one_folder)

        total_files_from = np.concatenate([total_files_from, files_corr_from])
        total_files_to = np.concatenate([total_files_to,files_corr_to])

        print('Size of saved lists : {}MB'.format((sys.getsizeof(total_files_from)+sys.getsizeof(total_files_to))/1e6))

        del segmentation_froms, semantic_froms, correspondances, \
            from_correct_xs, from_correct_ys, to_correct_xs, to_correct_ys, \
            from_wrong_object_xs, from_wrong_object_ys, to_wrong_object_xs, to_wrong_object_ys, \
            from_wrong_xs, from_wrong_ys, to_wrong_xs, to_wrong_ys, files_corr_from, files_corr_to

        gc.collect()

    torch.save(total_files_from, write_folder_base + "/tensors/full_f_from")
    torch.save(total_files_to, write_folder_base + "/tensors/full_f_to")