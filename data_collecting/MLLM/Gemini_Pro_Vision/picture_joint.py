import matplotlib.pyplot as plt
import numpy as np
import skimage as io
from skimage import io, transform, img_as_ubyte

width = 800

def normalization(pic):
    if pic.shape[0] != width:
        picture = transform.resize(pic, (800, 800))
        picture = img_as_ubyte(picture)
    else:
        picture = pic
    return picture


def picture_joint(pic1_path, pic2_path, pic3_path):
    pic1 = io.imread(pic1_path)
    pic2 = io.imread(pic2_path)
    pic3 = io.imread(pic3_path)

    pic1 = normalization(pic1)
    pic2 = normalization(pic2)
    pic3 = normalization(pic3)

    result = np.zeros((width+20, 3 * width + 220, 3))+255  
    result[10:width + 10, 10: width+10, :] = pic1.copy()
    result[10:width + 10, width + 110: 2*width + 110, :] = pic2.copy()
    result[10:width + 10, 2*width + 210:3*width + 210, :] = pic3.copy()

    result = np.array(result, dtype=np.uint8)  

    # plt.imshow(result)  
    # plt.show()
    # io.imsave(result_path, result)
    
    return result
