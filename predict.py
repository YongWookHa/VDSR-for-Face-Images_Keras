import sys
import os
from keras.models import model_from_json
import numpy as np
from scipy.misc import toimage, imread, imresize

from score_images import score

if __name__ == '__main__':
    json_path = sys.argv[1]
    w_path = sys.argv[2]
    img_path = sys.argv[3]
    dst_path = sys.argv[4]
    target_size = (128, 128) # should be changed according to size of output image
    print("--------------------------------")
    print('json_path : ', json_path)
    print('w_path : ', w_path)
    print('img_path : ', img_path)
    print('dst_path : ', dst_path)
    print("--------------------------------")

    with open(json_path, 'r') as f:
        vdsr = model_from_json(f.read())
    vdsr.load_weights(w_path)

    li = os.listdir(img_path)

    target_path = '%s/%s/' % (img_path, dst_path)
    os.makedirs(target_path, exist_ok=True)
    for filename in li:
        if filename[-4:] == '.jpg':
            img = imread(os.path.join(img_path, filename))
            img = imresize(img, target_size, interp='bicubic')
            img = np.array(img) / 127.5 - 1.
            img = img.reshape((1,)+target_size+(3,))  
            img = vdsr.predict(img)
            print(filename)
            img = img.reshape(target_size+(3,))
            img = (0.5 * img + 0.5) * 255
            toimage(img, cmin=0.0, cmax=255).save('%s/%s' % (target_path, filename))
        else:
            pass

    score(target_path)