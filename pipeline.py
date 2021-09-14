import sys, os
import argparse
import numpy as np
import logging
import time
from data_utils.transformation import read_transform, camera_to_base
from data_utils import cut_ZividPCDandTransform

# Corners_1 = [(-460.03,340,240), (-741.7,340,240), (-741.7,170,240), (-460,170,240), (-460,340,5), (-741.7,340,5), (-741.7,170,5), (-460,170,5)]
Corners = [(35,880,300), (35,1150,300), (-150,1150,300), (-150,880,300), (35,880,50), (35,1150,50), (-150,1150,50), (-150,880,50)]
Cam_inBlensor_position = (-0.136411824,	-0.589879807, 4.067452035, 0.033196298,	0.144020323, -1.57)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
data_path = ROOT_DIR + '/data'

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Pipeline')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--log_dir', type=str, default='pointnet2_sem_seg_finetune', help='experiment root')
    parser.add_argument('--root', type=str, default=data_path + '/Test_set/', help='file need to be tested')
    parser.add_argument('--save_img', action='store_true', default=False, help='save the point cloud as image [default: False]')
    return parser.parse_args()


'''load transformation matrix '''
transforMatrix_path = ROOT_DIR + '/data_utils/meta/transformation_2.yaml'
transform_matrix = read_transform(transforMatrix_path)

'''data prepareration folder and final output folder '''
if not os.path.exists(ROOT_DIR + '/final_output'):
    os.makedirs(ROOT_DIR+ '/final_output')
else:
    pass
if not os.path.exists(ROOT_DIR + '/seg_results'):
    os.makedirs(ROOT_DIR+ '/seg_results')
else:
    os.system('rm -r %s' % ROOT_DIR + '/seg_results')
    os.makedirs(ROOT_DIR+ '/seg_results')
if not os.path.exists(data_path + '/Test_set'):
    os.makedirs(data_path + '/Test_set')
else: 
    os.system('rm -r %s' % data_path+'/Test_set')
    os.makedirs(data_path + '/Test_set')
if not os.path.exists(data_path + '/res_scene'):
    os.makedirs(data_path + '/res_scene')
else:
    os.system('rm -r %s' % data_path+'/res_scene')
    os.makedirs(data_path + '/res_scene')
time_=time.strftime("%Y-%m-%d-%H-%M-%S")
outdir = ROOT_DIR + '/final_output/'
mytimedir = outdir + time_ +"/"
if not os.path.exists(mytimedir):
    os.makedirs(mytimedir)
else:
    pass


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
   # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'final_output/'

    '''LOG'''
    logger = logging.getLogger("Pipeline")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    file_handler = logging.FileHandler('%s/pipeline_log.txt' % mytimedir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('Pipeline start ...')

    '''main project start'''
    List_WholeScene = []
    List_zivid = os.listdir(data_path)
    for index in List_zivid:
        if os.path.splitext(index)[1] == '.pcd':
            List_WholeScene.append(index)
    k = 1
    '''prepare data:
                    cut the whole scene into cuboid
                    transform the corordinate in cuboid
                    save the transformed cuboid and remaining part
    '''
    log_string('Start data prepareration process: The number of data is %d' % (len(List_WholeScene)))
    for scene in List_WholeScene:
        pcd_path = data_path + '/' + scene
        os.system('cp '+ pcd_path + ' ' + mytimedir)
        whole_scene = cut_ZividPCDandTransform.Read_PCD(pcd_path)   # base open3d
        if args.save_img:
            cut_ZividPCDandTransform.save_pcd_asIMG(whole_scene, mytimedir + scene.split('.pcd')[0] + '.png')
        patch_motor = cut_ZividPCDandTransform.cut_motorPatch(Corners, transform_matrix, whole_scene)
        patch_scene = cut_ZividPCDandTransform.cut_scenePatch(Corners, transform_matrix, whole_scene)
        new_cor = cut_ZividPCDandTransform.transform_patch(patch_motor, transform_matrix, Cam_inBlensor_position)
       # break
        np_name = scene.split(".")[0]
        np.save(data_path + '/Test_set/' + 'Test_' + np_name, new_cor)
        np.save(data_path + '/res_scene/' + 'Res_' + np_name, patch_scene)
        log_string("Cutting process of zivid data: %s -------------> number: %s is finished" %(scene,k))
        k += 1
    log_string('Data prepareration process is finished! Now start segmentation by PointNet++ ...')

    '''run PointNet++ model'''
    GPU = args.gpu
   
    os.system("CUDA_VISIBLE_DEVICES=%s python pipeline_seg.py --log_dir %s --root %s --visual" % (GPU, args.log_dir, args.root))
    log_string("Segmentation by PointNet++ is Done!")
            

    seg_results = []
    seg_results_dir = os.listdir(ROOT_DIR + '/seg_results/')
    for result in seg_results_dir :
        if os.path.splitext(result)[1] == '.npy':
            seg_results.append(result)
        elif os.path.splitext(result)[1] == '.txt':
            os.system('cp '+ ROOT_DIR+ '/seg_results/'+ result + ' '+mytimedir)
    k = 1
    for seg_result in seg_results:
        npy_path = ROOT_DIR + '/seg_results/' + seg_result
        seg_cor = np.load(npy_path)
        seg_motor = cut_ZividPCDandTransform.retransform_patch(seg_cor, transform_matrix, Cam_inBlensor_position)
        
        '''find bolts'''
        bolt_locations, num_bolt = cut_ZividPCDandTransform.find_bolts(seg_motor, eps=2, min_points=50)
        log_string('number of bolt from %s is: %d ' % (seg_result, num_bolt))
        log_string('Each location of them based on Zivid-coordinate is: ------------------------------')
        for bolt_location in bolt_locations:
            log_string('({},{},{})'.format(bolt_location[0], bolt_location[1], bolt_location[2]))
        # bolt_side_po_zivid = np.argmin(bolt_locations, axis = 0)[2]
        # bolt_side_zivid = bolt_locations[bolt_side_po_zivid]

        log_string('Each location of them based on Robot-coordinate is: ------------------------------')
        min_Z = bolt_locations[0]
        for bolt_location in bolt_locations:
            bolt_location_robot= camera_to_base(transform_matrix, bolt_location[0:3])
            log_string('({},{},{})'.format(bolt_location_robot[0], bolt_location_robot[1], bolt_location_robot[2]))
            if bolt_location_robot[2] < min_Z[2]: min_Z = bolt_location_robot

        log_string('The bolt on the side of motor based on Robot-coordinate is: ({},{},{})'.format(min_Z[0],min_Z[1],min_Z[2]))

        print('Start the insert process')
        '''insert process'''
        for res_scene_dir in os.listdir(data_path + '/res_scene/') :
            if res_scene_dir.split('Res_')[1] == seg_result.split('Test_')[1]:
                res_scene = np.load(data_path + '/res_scene/' + res_scene_dir)
                res_scene[:, 3:] = [0,0,255]  # change the remaining scene color into blue
                final_result = np.vstack([seg_motor, res_scene])
               # np.save(mytimedir + res_scene_dir.split('Res_')[1], final_result)
                cut_ZividPCDandTransform.save_pcd(final_result, mytimedir + res_scene_dir.split('Res_')[1].split('.npy')[0]+'_segResult')
                if args.save_img:
                    cut_ZividPCDandTransform.save_pcd_asIMG(mytimedir + res_scene_dir.split('Res_')[1].split('.npy')[0]+'_segResult.png')
                print("insert process of zivid data: %s -------------> number: %s is finished" %(res_scene_dir.split('Res_')[1],k))

        k += 1
    os.system('rm -r %s' % ROOT_DIR + '/seg_results')
    print('all done!')


if __name__ == '__main__' :
    args = parse_args()
    main(args)