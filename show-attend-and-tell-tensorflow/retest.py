import matplotlib.pyplot as plt
import cPickle as pickle
import tensorflow as tf
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
from core.bleu import evaluate

#%matplotlib inline
plt.rcParams['figure.figsize'] = (8.0, 6.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#%load_ext autoreload
#%autoreload 2
def main():
    data = load_coco_data(data_path='./data', split='val')
    with open('./data/train/word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)
        print data
        model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                                 dim_hidden=1024, n_time_step=16, prev2out=True,
                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
        solver = CaptioningSolver(model, data, data, n_epochs=20, batch_size=128, update_rule='adam',
                                  learning_rate=0.001, print_every=1000, save_every=1, image_path='./image/val2014_resized',
                                  pretrained_model=None, model_path='model/lstmval/', test_model='model/lstm/model-10',
                                  print_bleu=True, log_path='log/')

        #solver.test(data, split='val')
        solver.test(data, split='test')
if __name__ == "__main__":
    main()