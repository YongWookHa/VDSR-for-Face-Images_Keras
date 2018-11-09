import os
import numpy as np
import scipy.misc

from inception_score import get_inception_score

def score(path):
    print("--------------------------------")
    print('path:', path)
    print("--------------------------------")

    li = os.listdir(path)
    imgs = []
    for filename in li:
        if filename[-4:] == '.jpg':
            img = scipy.misc.imread(os.path.join(path, filename))
            imgs.append(img)
        else:
            pass

    print('min : ', np.min(imgs[0]))
    print('max : ', np.max(imgs[0]))

    result = get_inception_score(imgs)

    with open('%s/inception_score.txt' % path, 'w') as f:
        f.write('inception score...\n')
        f.write('mean : %s\n' % result[0])
        f.write('std : %s\n' % result[1])
        f.close()

    print('scoring finished!')
    print('mean : %s\n' % result[0])
    print('std : %s\n' % result[1])
