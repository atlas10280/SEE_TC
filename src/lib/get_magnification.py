import nd2

def get_magnification(nd2_path):
    with nd2.ND2File(nd2_path) as ndfile:
        text_unparsed = ndfile.text_info
        optics = text_unparsed['optics']

    ########### check the magnification
    if '10x' in optics:
            input_magnification = 10       
            return(input_magnification)                 
    elif '20x' in optics:
            input_magnification = 20
            return(input_magnification)
    elif '40x' in optics:
            input_magnification = 40
            return(input_magnification)
    else:
            print("skipping sample, unknown magnification")
    