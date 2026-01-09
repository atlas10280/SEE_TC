import nd2

def extract_image_3Channel(file, c1, c2, c3):    
    print("Extracting image array:")
    print(file)
    img = nd2.imread(file)
    with nd2.ND2File(file) as m:     
        c_names = m._channel_names
        idx_1 = [i for i, s in enumerate(c_names) if c1 in s]
        idx_2 = [i for i, s in enumerate(c_names) if c2 in s]
        idx_3 = [i for i, s in enumerate(c_names) if c3 in s]
        img = img[(int(idx_1[0]),int(idx_2[0]),int(idx_3[0])),:,:]
        return img