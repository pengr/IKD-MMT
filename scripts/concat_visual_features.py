import numpy as np

visual_dir =  '/home/think/data/multi30k/features_resnet50/'
train = visual_dir+ 'train-resnet50-res4frelu.npy-002'
valid = visual_dir+ 'val-resnet50-res4frelu.npy'
test = visual_dir+ 'test_2016_flickr-resnet50-res4frelu.npy'
test1 = visual_dir+ 'test_2017_coco-resnet50-res4frelu.npy'
test2 = visual_dir+ 'test_2017_flickr-resnet50-res4frelu.npy'
all = visual_dir+ 'resnet50-res4frelu.npy'


def main():
    train_arr = np.load(train)
    valid_arr = np.load(valid)
    test_arr = np.load(test)
    test1_arr = np.load(test1)
    test2_arr = np.load(test2)
    all_arr = np.concatenate((train_arr, valid_arr, test_arr, test1_arr, test2_arr), axis=0)
    np.save(all, all_arr)


if "__main__" == __name__:
    main()
