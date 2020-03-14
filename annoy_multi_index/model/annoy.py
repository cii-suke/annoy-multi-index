import os
import numpy as np
from annoy import AnnoyIndex
from dataset import load
from tqdm import tqdm

category_len = 10

class _Annoy(object):

    def __init__(self, feature):
        model_path = 'model.ann'.format()
        n_dim = feature.shape[1] * feature.shape[2]
        feature = feature.reshape(feature.shape[0], n_dim)
        self.t = AnnoyIndex(n_dim, 'angular')
        if not os.path.exists(model_path):
            for i, f in enumerate(tqdm(feature)):
                # normarize
                v = f / np.sum(f)
                self.t.add_item(i, v)
            self.t.build(10)
            self.t.save(model_path)
        else:
            self.t.load(model_path)
    


class Annoy(object):

    def __init__(self):
        d = load.DataLoader(train=True)
        self.x, self.y, self.x_ori = d.load()
        a = _Annoy(self.x)
        self.models = a.t
        # self.models = []
        # for i in range(category_len):
        #     self.models.append(_Annoy(i))

    def pred(self, target):
        # t = self.models[idx]
        t = self.models
        target = target.reshape((target.shape[0] * target.shape[1]))
        target = target / np.sum(target)
        index, distance = t.get_nns_by_vector(target, 5, search_k=-1, include_distances=True)
        return self.x_ori[index], self.y[index]