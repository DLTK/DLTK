import numpy as np

def extract_class_balanced_example_array(image, label, example_size=[1,64,64], n_examples=1, n_classes=2):
    
    """
        Extract training examples from an image (and corresponding label) subject to class balancing.
        Returns an image example array and the corresponding label array.
        
        Parameters
        ----------
        image: float
            3D numpy array
        label: 
            3D numpy array or None
        example_size: 
            list of int
        n_examples:
            int
            
        Returns
        -------
        ex_imgs, ex_lbls 
            tuple of 4D numpy arrays, where dim 0 is the dimension of examples
    """
    
    assert isinstance(image, (np.ndarray, np.generic) )
    assert isinstance(label, (np.ndarray, np.generic) )
    assert len(image.shape) == 5
    assert image.shape[0] == 1
    assert np.allclose(image.shape[:-1], label.shape[:-1]) 
    assert len(example_size) == 3
    assert n_classes > 1
    assert n_examples >= n_classes
    
    n_ex_per_class = np.round(n_examples/n_classes)
    
    # compute an example radius as we are extracting centered around locations
    ex_rad = np.array(list(zip(np.floor(np.array(example_size)/2.0), np.ceil(np.array(example_size)/2.0))), dtype=np.int)

    ex_imgs = []
    ex_lbls = []   
    for c in range(n_classes):
        # get valid, random center locations belonging to that class
        idx_5d = np.argwhere(label == c)
        idx = idx_5d[:,1:4]
        
        # if a class is not available, extract random examples (FFS, need a fixed batch size for the damn multicore batcher)
        if len(idx) < n_ex_per_class:
            n_missing = n_ex_per_class - len(idx)
            valid_loc_range = [[ex_rad[0][0], image.shape[1]-ex_rad[0][1]],
                               [ex_rad[1][0], image.shape[2]-ex_rad[1][1]],
                               [ex_rad[2][0], image.shape[3]-ex_rad[2][1]]]
            rx = np.random.randint(valid_loc_range[0][0], valid_loc_range[0][1], size=n_missing)
            ry = np.random.randint(valid_loc_range[1][0], valid_loc_range[1][1], size=n_missing) 
            rz = np.random.randint(valid_loc_range[2][0], valid_loc_range[2][1], size=n_missing) 
            
            missing_idx = np.array([list(a) for a in zip(rx,ry,rz)])            
            idx = missing_idx if len(idx) == 0 else idx + missing_idx
            
        # extract random locations
        r_idx = idx[np.random.randint(len(idx), size=n_ex_per_class, dtype=np.int64), :]
        
        # add a random shift them to avoid learning a centre bias - IS THIS REALLY TRUE?
        r_shift_x = np.random.randint(-ex_rad[0][0], ex_rad[0][1], size=n_ex_per_class)
        r_shift_y = np.random.randint(-ex_rad[1][0], ex_rad[1][1], size=n_ex_per_class)
        r_shift_z = np.random.randint(-ex_rad[2][0], ex_rad[2][1], size=n_ex_per_class)
        r_shift = np.array([list(a) for a in zip(r_shift_x,r_shift_y,r_shift_z)])        
        
        r_idx += r_shift
        
        
        # shift them to valid locations if necessary
        for r in r_idx:
            r[0] = max(min(r[0], image.shape[1] - ex_rad[0][1]), ex_rad[0][0])
            r[1] = max(min(r[1], image.shape[2] - ex_rad[1][1]), ex_rad[1][0])
            r[2] = max(min(r[2], image.shape[3] - ex_rad[2][1]), ex_rad[2][0])
    
        for i in range(len(r_idx)):
            # extract class-balanced examples from the original image
            ex_img = image[0:, 
                           r_idx[i][0]-ex_rad[0][0]:r_idx[i][0]+ex_rad[0][1],
                           r_idx[i][1]-ex_rad[1][0]:r_idx[i][1]+ex_rad[1][1],
                           r_idx[i][2]-ex_rad[2][0]:r_idx[i][2]+ex_rad[2][1],
                           0:]
            
            ex_lbl = label[0:,
                           r_idx[i][0]-ex_rad[0][0]:r_idx[i][0]+ex_rad[0][1],
                           r_idx[i][1]-ex_rad[1][0]:r_idx[i][1]+ex_rad[1][1],
                           r_idx[i][2]-ex_rad[2][0]:r_idx[i][2]+ex_rad[2][1],
                           0:]
            
            # zero pad the example if necessary:
            padding_3d = np.array(example_size) - ex_img.shape[1:4]
            
            if np.sum(padding_3d) > 0:
                padding_5d = list(zip([0,0,0,0,0], np.insert(padding_3d, [0, 3], [0, 0])))
                ex_img = np.pad(ex_img, padding_5d, 'constant', constant_values=[[0,0],[0,0],[0,0],[0,0],[0,0]]) 
                ex_lbl = np.pad(ex_lbl, padding_5d, 'constant', constant_values=[[0,0],[0,0],[0,0],[0,0],[0,0]])
           
            # concatenate and return the 4D examples  
            ex_imgs = np.concatenate((ex_imgs, ex_img), axis=0) if (len(ex_imgs) != 0) else ex_img
            ex_lbls = np.concatenate((ex_lbls, ex_lbl), axis=0) if (len(ex_lbls) != 0) else ex_lbl

    assert list(ex_imgs.shape) == [n_examples, example_size[0], example_size[1], example_size[2], image.shape[-1]]
    assert list(ex_lbls.shape) == [n_examples, example_size[0], example_size[1], example_size[2], 1]
      
    return ex_imgs.astype(np.float32), ex_lbls.astype(np.int32)


def extract_class_balanced_example_scalar(image, label, example_size=[1,64,64], n_examples=1, n_classes=2):
    
    """
        Extract training examples from an image (and corresponding label) array subject to class balancing. 
        Returns an image example array and the corresponding class as a scalar int.

        Parameters
        ----------
        image: float
            5D numpy array
        label: 
            5D numpy array
        example_size: 
            list of int
        n_examples:
            int
            
        Returns
        -------
        ex_imgs, ex_lbls 
            tuple of 5D and 2D numpy arrays, where dim 0 is the dimension of examples
    """
    
    assert isinstance(image, (np.ndarray, np.generic) )
    assert isinstance(label, (np.ndarray, np.generic) )
    assert len(image.shape) == 5
    assert np.allclose(image.shape[:-1], label.shape[:-1]) 
    assert len(example_size) == 3
    assert n_classes > 1
    assert n_examples >= n_classes
    
    n_ex_per_class = np.round(n_examples/n_classes)
    
    # compute an example radius as we are extracting centered around locations
    ex_rad = np.array(list(zip(np.floor(np.array(example_size)/2.0), np.ceil(np.array(example_size)/2.0))), dtype=np.int)
    
    valid_loc_range = [[ex_rad[0][0], image.shape[1]-ex_rad[0][1]],
                       [ex_rad[1][0], image.shape[2]-ex_rad[1][1]],
                       [ex_rad[2][0], image.shape[3]-ex_rad[2][1]]]

    ex_imgs = []
    ex_lbls = []   
    for c in list(range(1, n_classes))+[0]:
        # random center locations belonging to that class
        idx_5d = np.argwhere(label == c)
        idx = idx_5d[:,1:4]
        
        # extract valid random locations; if a class is not available, extract additional samples from the background instead
        if len(idx) == 0:
            continue        
        if c==0:
            r_idx_idx = np.random.choice(len(idx), size=n_examples-len(ex_imgs), replace=False)
            #r_idx_idx = np.random.randint(len(idx), size=n_examples-ex_imgs.shape[0], dtype=np.int64)
        else:
            r_idx_idx = np.random.choice(len(idx), size=min(n_ex_per_class,len(idx)), replace=False)

        r_idx = idx[r_idx_idx, :]
        
        for i in range(len(r_idx)):
            # extract class-balanced examples from the original image
            ex0 = max(r_idx[i][0]-ex_rad[0][0],0)
            ey0 = max(r_idx[i][1]-ex_rad[1][0],0)
            ez0 = max(r_idx[i][2]-ex_rad[2][0],0)
            
            ex1 = min(r_idx[i][0]+ex_rad[0][1],image.shape[1])
            ey1 = min(r_idx[i][1]+ex_rad[1][1],image.shape[2])
            ez1 = min(r_idx[i][2]+ex_rad[2][1],image.shape[3])
            
            ex_img = image[0:, ex0:ex1, ey0:ey1, ez0:ez1, 0:]
            
            ex_lbl = np.expand_dims(np.asarray([c]), axis=0)
            
            # zero pad the example if necessary:
            padding_3d = np.array(example_size) - ex_img.shape[1:4]

            if np.sum(padding_3d) > 0:
                padding_5d = list(zip([0,0,0,0,0], np.insert(padding_3d, [0, 3], [0, 0])))
                ex_img = np.pad(ex_img, padding_5d, 'constant', constant_values=[[0,0],[0,0],[0,0],[0,0],[0,0]])
            
            # concatenate and return the 4D examples  
            ex_imgs = np.concatenate((ex_imgs, ex_img), axis=0) if (len(ex_imgs) != 0) else ex_img
            ex_lbls = np.concatenate((ex_lbls, ex_lbl), axis=0) if (len(ex_lbls) != 0) else ex_lbl
            
    assert list(ex_imgs.shape) == [n_examples, example_size[0], example_size[1], example_size[2], image.shape[-1]]
    assert list(ex_lbls.shape) == [n_examples, 1]

    return ex_imgs.astype(np.float32), ex_lbls.astype(np.int32)


def extract_random_example_array(image, label=np.array([]), example_size=[1,64,64], n_examples=1):
    
    """
        Randomly extract training examples from image (and corresponding label).
        Returns an image example array and the corresponding label array.

        Parameters
        ----------
        image: float
            5D numpy array
        label: 
            5D numpy array or None
        example_size: 
            list of int
        n_examples:
            int
            
        Returns
        -------
        ex_imgs, ex_lbls 
            tuple of 5D numpy arrays, where dim 0 is the dimension of examples
    """
    
    assert isinstance(image, (np.ndarray, np.generic) )
    assert isinstance(label, (np.ndarray, np.generic) )
    assert len(image.shape) == 5
    if label.size:
        assert np.allclose(image.shape[:-1], label.shape[:-1]) 
    assert len(example_size) == 3
    assert n_examples > 0
    
    # extract random examples from image and label
    valid_loc_range = [image.shape[1]-example_size[0],
                       image.shape[2]-example_size[1],
                       image.shape[3]-example_size[2]]
    
    rx = np.random.randint(valid_loc_range[0], size=n_examples) if valid_loc_range[0] > 0 else np.zeros(n_examples, dtype=int)
    ry = np.random.randint(valid_loc_range[1], size=n_examples) if valid_loc_range[1] > 0 else np.zeros(n_examples, dtype=int)
    rz = np.random.randint(valid_loc_range[2], size=n_examples) if valid_loc_range[2] > 0 else np.zeros(n_examples, dtype=int)
        
    rnd_loc = [rx,ry,rz]
        
    ex_imgs = []
    ex_lbls = []
    for i in range(n_examples):
        
        # extract randomly cropped examples from the original image
        ex_img = image[0:,
                       rnd_loc[0][i]:rnd_loc[0][i]+example_size[0],
                       rnd_loc[1][i]:rnd_loc[1][i]+example_size[1],
                       rnd_loc[2][i]:rnd_loc[2][i]+example_size[2],
                       0:]

        if label.size: 
            ex_lbl = label[0:,
                           rnd_loc[0][i]:rnd_loc[0][i]+example_size[0],
                           rnd_loc[1][i]:rnd_loc[1][i]+example_size[1],
                           rnd_loc[2][i]:rnd_loc[2][i]+example_size[2],
                           0:]

        # zero pad the example if necessary:
        padding_3d = np.array(example_size) - ex_img.shape[1:4]

        if np.sum(padding_3d) > 0:
            padding_5d = list(zip([0,0,0,0,0], np.insert(padding_3d, [0, 3], [0, 0])))
            ex_img = np.pad(ex_img, padding_5d, 'constant', constant_values=[[0,0],[0,0],[0,0],[0,0],[0,0]]) 
            if label.size:
                ex_lbl = np.pad(ex_lbl, padding_5d, 'constant', constant_values=[[0,0],[0,0],[0,0],[0,0],[0,0]])
        
        # concatenate and return the examples  
        ex_imgs = np.concatenate((ex_imgs, ex_img), axis=0) if (len(ex_imgs) != 0) else ex_img
        
        if label.size:
            ex_lbls = np.concatenate((ex_lbls, ex_lbl), axis=0) if (len(ex_lbls) != 0) else ex_lbl
        
    assert list(ex_imgs.shape) == [n_examples, example_size[0], example_size[1], example_size[2], image.shape[-1]]
    if label.size:
        assert list(ex_lbls.shape) == [n_examples, example_size[0], example_size[1], example_size[2], 1]

    return np.array(ex_imgs, dtype=np.float32), np.array(ex_lbls, dtype=np.int32)


def auto_pad_crop_full_volumes_array(image, img_size=[64,64,64], pad_mode='zero'):
    """
            Randomly extract training examples from image.
            Returns an image example array.

            Parameters
            ----------
            image: float
                5D numpy array
            example_size:
                list of int
            n_examples:
                int

            Returns
            -------
            ex_imgs
                tuple of 5D numpy arrays, where dim 0 is the dimension of examples
        """

    assert isinstance(image, (np.ndarray, np.generic))
    assert len(image.shape) == 5
    assert len(img_size) == 3
    assert(pad_mode in ('zero', 'min_val', 'none'))

    from_indices = [[0, image.shape[1]], [0, image.shape[2]], [0, image.shape[3]]]
    to_indices = [[0, img_size[0]], [0, img_size[1]], [0, img_size[2]]]

    for i in range(3):
        if image.shape[i + 1] <= img_size[i]:
            to_indices[i][0] = int(np.floor((img_size[i] - image.shape[i + 1]) / 2.))
            to_indices[i][1] = to_indices[i][0] + image.shape[i + 1]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i + 1] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

    if pad_mode == 'zero':
        tmp_img = np.zeros(image.shape[:1] + tuple(img_size) + image.shape[-1:])
    elif pad_mode == 'min_val':
        min_pix = np.min(image)
        tmp_img = np.ones(image.shape[:1] + tuple(img_size) + image.shape[-1:]) * min_pix  # assume background is not 0 but min_pix value
    if pad_mode == 'none':
        ex_imgs = np.asarray(image)
    else:
        tmp_img[:,
                to_indices[0][0]:to_indices[0][1],
                to_indices[1][0]:to_indices[1][1],
                to_indices[2][0]:to_indices[2][1]] = image[:,
                                                           from_indices[0][0]:from_indices[0][1],
                                                           from_indices[1][0]:from_indices[1][1],
                                                           from_indices[2][0]:from_indices[2][1]]
        ex_imgs = np.asarray(tmp_img)
    return ex_imgs.astype(np.float32)

