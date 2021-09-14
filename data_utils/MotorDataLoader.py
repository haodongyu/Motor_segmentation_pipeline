import os
import numpy as np
from numpy.random import choice
from tqdm import tqdm
from torch.utils.data import Dataset

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
   # print('center of this point cloud is:', centroid)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def pc_denormalize(pc_normalized, centroid, m):
    pc = m*pc_normalized + centroid
    return pc


class ScannetDatasetwholeMotor():  # for test
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area='Validation', block_size=50.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.test_area = test_area
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('%s' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('%s' % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.motor_coord_min, self.motor_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])    # num_files*num_points*6
            self.semantic_labels_list.append(data[:, 6])

            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.motor_coord_min.append(coord_min), self.motor_coord_max.append(coord_max)

        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(6)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(7))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
       # print('size of init_points[]', point_set_ini.shape)
        points = point_set_ini[:,:6] 
       # print('size of points[]', points.shape)
        labels_o = self.semantic_labels_list[index]
       # coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        N_points = points.shape[0]
        '''
        under Test: coor_min = [-255.2752533  -110.75037384  519.2489624 ]
                    coor_max = [ 45.58950043 125.44611359 797.55413818]
        under Validation: coor_min = [-0.22853388 -0.6627815  -3.93524766]
                          coor_max = [ 0.76870452  0.74199943 -2.73275002]
        '''
        normalized_motor = pc_normalize(points[:, :3])
       # normalized_motor = np.concatenate((normalized_motor, points[:, 3:]), axis=1)  # num_point * 6
        normalized_motor = np.hstack((normalized_motor, labels_o.reshape(len(labels_o), 1))) # num_point * 4 / 7
        np.random.shuffle(normalized_motor)
        labels = normalized_motor[:, -1]
        normalized_motor = normalized_motor[:,:3]


        # normalize  with RGB
       # current_points = np.concatenate((normalized_motor, points[:, 3:]), axis=1)  # num_point * 6


        ### pad 0 into last block, which not enough to 4096 ###
        num_block= divmod(N_points, self.block_points)  # num and res
        # data_motor = np.zeros((110, 4096, 9))
        # data_label = np.zeros((110, 4096))
        data_motor = normalized_motor[0 : num_block[0]*self.block_points, :].reshape((num_block[0], self.block_points, normalized_motor.shape[1]))  # num_block*N*3
        data_label = labels[0:num_block[0]*self.block_points].reshape(-1, self.block_points)
        if num_block[1]:
            block_res = np.zeros((self.block_points, 3))
            label_res = np.zeros(self.block_points)
            block_res[0:num_block[1], :] = normalized_motor[N_points-num_block[1]:, :]
            label_res[0:num_block[1]] = labels[N_points-num_block[1]:]
            block_res = np.array([block_res])
            data_motor = np.vstack([data_motor, block_res])
            data_label = np.vstack([data_label, label_res])
       # labels = labels.reshape((-1, self.block_points))  # num_block*N
    
       # print('current data size after normalized: ', data_motor[0].shape)
        return data_motor, data_label


    def __len__(self):
        return len(self.scene_points_list)


class ScannetDatasetNoLabelMotor():  # for test
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area='Validation', block_size=50.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.test_area = test_area
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('%s' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('%s' % test_area) is not -1]
        self.scene_points_list = []

        self.motor_coord_min, self.motor_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])    # num_files*num_points*6


            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.motor_coord_min.append(coord_min), self.motor_coord_max.append(coord_max)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
 
        N_points = points.shape[0]
        '''
        under Test: coor_min = [-255.2752533  -110.75037384  519.2489624 ]
                    coor_max = [ 45.58950043 125.44611359 797.55413818]
        '''
        normalized_motor, centroid, m = pc_normalize(points[:, :3])
        np.random.shuffle(normalized_motor)
       # normalized_motor = normalized_motor[:,:3]


        ### pad 0 into last block, which not enough to 4096 ###
        num_block = divmod(N_points, self.block_points)  # num and res
        data_motor = normalized_motor[0 : num_block[0]*self.block_points, :].reshape((num_block[0], self.block_points, normalized_motor.shape[1]))  # num_block*N*3
        if num_block[1]:
            block_res = np.zeros((self.block_points, 3))
            block_res[0:num_block[1], :] = normalized_motor[N_points-num_block[1]:, :]
            block_res = np.array([block_res])
            data_motor = np.vstack([data_motor, block_res])
    
       # print('current data size after normalized: ', data_motor[0].shape)
        return data_motor, centroid, m


    def __len__(self):
        return len(self.scene_points_list)


# if __name__ == '__main__':
    # data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    # num_point, test_area, block_size, sample_rate = 4096, 'Validation', 1.0, 0.01

    # point_data = MotorDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    # print('point data size:', point_data.__len__())
    # print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    # print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    # import torch, time, random
    # manual_seed = 123
    # random.seed(manual_seed)
    # np.random.seed(manual_seed)
    # torch.manual_seed(manual_seed)
    # torch.cuda.manual_seed_all(manual_seed)
    # def worker_init_fn(worker_id):
    #     random.seed(manual_seed + worker_id)
    # train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    # for idx in range(4):
    #     end = time.time()
    #     for i, (input, target) in enumerate(train_loader):
    #         print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
    #         end = time.time()
