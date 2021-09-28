import os
import numpy as np
import open3d as o3d
import csv
import math
from . import transformation
from sklearn.cluster import DBSCAN
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

def get_Corordinate_inCam(cam_pos_x, cam_pos_y, cam_pos_z, alpha, beta, theta, cor):

    alpha = float(alpha)
    beta = float(beta)
    theta = float(theta)
    cor = np.array(cor).T
    cam_pos = np.array([float(cam_pos_x), float(cam_pos_y), float(cam_pos_z)]).T
    cor = cor - cam_pos

    c_mw = np.array([[math.cos(beta)*math.cos(theta), math.cos(beta)*math.sin(theta), -math.sin(beta)],
            [-math.cos(alpha)*math.sin(theta)+math.sin(alpha)*math.sin(beta)*math.cos(theta), math.cos(alpha)*math.cos(theta)+math.sin(alpha)*math.sin(beta)*math.sin(theta), math.sin(alpha)*math.cos(beta)],
            [math.sin(alpha)*math.sin(theta)+math.cos(alpha)*math.sin(beta)*math.cos(theta), -math.sin(alpha)*math.cos(theta)+math.cos(alpha)*math.sin(beta)*math.sin(theta), math.cos(alpha)*math.cos(beta)] ])

    cor_new = c_mw.dot(cor)

    return cor_new

def get_Corordinate_inBlensor_rw(cam_pos_x, cam_pos_y, cam_pos_z, alpha, beta, theta, cor_new):
    cor_inBlensor_Cam = cor_new
    alpha = float(alpha)
    beta = float(beta)
    theta = float(theta)
    cam_pos = np.array([float(cam_pos_x), float(cam_pos_y), float(cam_pos_z)]).T

    c_mw = np.array([[math.cos(beta)*math.cos(theta), math.cos(beta)*math.sin(theta), -math.sin(beta)],
            [-math.cos(alpha)*math.sin(theta)+math.sin(alpha)*math.sin(beta)*math.cos(theta), math.cos(alpha)*math.cos(theta)+math.sin(alpha)*math.sin(beta)*math.sin(theta), math.sin(alpha)*math.cos(beta)],
            [math.sin(alpha)*math.sin(theta)+math.cos(alpha)*math.sin(beta)*math.cos(theta), -math.sin(alpha)*math.cos(theta)+math.cos(alpha)*math.sin(beta)*math.sin(theta), math.cos(alpha)*math.cos(beta)] ])
    c_mw_I = np.linalg.inv(c_mw)
    cor = c_mw_I.dot(cor_inBlensor_Cam) + cam_pos
    return cor

def get_panel(point_1, point_2, point_3):

    x1 = point_1[0]
    y1 = point_1[1]
    z1 = point_1[2]

    x2 = point_2[0]
    y2 = point_2[1]
    z2 = point_2[2] 

    x3 = point_3[0]
    y3 = point_3[1]
    z3 = point_3[2]
    
    a = (y2-y1)*(z3-z1) - (y3-y1)*(z2-z1)
    b = (z2-z1)*(x3-x1) - (z3-z1)*(x2-x1)
    c = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    d = 0 - (a*x1 + b*y1 + c*z1)

    return (a, b, c, d)


def set_Boundingbox(panel_list, point_cor):

    if panel_list['panel_up'][0]*point_cor[0] + panel_list['panel_up'][1]*point_cor[1] + panel_list['panel_up'][2]*point_cor[2] + panel_list['panel_up'][3] <= 0 :   # panel 1
        if panel_list['panel_bot'][0]*point_cor[0] + panel_list['panel_bot'][1]*point_cor[1] + panel_list['panel_bot'][2]*point_cor[2] + panel_list['panel_bot'][3] >= 0 : # panel 2
            if panel_list['panel_front'][0]*point_cor[0] + panel_list['panel_front'][1]*point_cor[1] + panel_list['panel_front'][2]*point_cor[2] + panel_list['panel_front'][3] <= 0 : # panel 3
                if panel_list['panel_behind'][0]*point_cor[0] + panel_list['panel_behind'][1]*point_cor[1] + panel_list['panel_behind'][2]*point_cor[2] + panel_list['panel_behind'][3] >= 0 : # panel 4
                    if panel_list['panel_right'][0]*point_cor[0] + panel_list['panel_right'][1]*point_cor[1] + panel_list['panel_right'][2]*point_cor[2] + panel_list['panel_right'][3] >= 0 : #panel 5
                        if panel_list['panel_left'][0]*point_cor[0] + panel_list['panel_left'][1]*point_cor[1] + panel_list['panel_left'][2]*point_cor[2] + panel_list['panel_left'][3] >= 0 : # panel 6

                            return True
    return False

def cut_scenePatch(Corners, cam_to_robot_transform, whole_scene):
    cor_inCam = []
    for corner in Corners:
        cor_inCam_point = transformation.base_to_camera(cam_to_robot_transform, np.array(corner))
        cor_inCam.append(np.squeeze(np.array(cor_inCam_point)))

    panel_1 = get_panel(cor_inCam[0], cor_inCam[1], cor_inCam[2])
    panel_2 = get_panel(cor_inCam[5], cor_inCam[6], cor_inCam[4])
    panel_3 = get_panel(cor_inCam[0], cor_inCam[3], cor_inCam[4])
    panel_4 = get_panel(cor_inCam[1], cor_inCam[2], cor_inCam[5])
    panel_5 = get_panel(cor_inCam[0], cor_inCam[1], cor_inCam[4])
    panel_6 = get_panel(cor_inCam[2], cor_inCam[3], cor_inCam[6])
    panel_list = {'panel_up':panel_1, 'panel_bot':panel_2, 'panel_front':panel_3, 'panel_behind':panel_4, 'panel_right':panel_5, 'panel_left':panel_6}

    patch_scene = []
    for point in whole_scene:
        point_cor = (point[0], point[1], point[2])
        if not set_Boundingbox(panel_list, point_cor):
            patch_scene.append(point)
    return patch_scene

def cut_motorPatch(Corners, cam_to_robot_transform, whole_scene):
    cor_inCam = []
    for corner in Corners:
        cor_inCam_point = transformation.base_to_camera(cam_to_robot_transform, np.array(corner))
        cor_inCam.append(np.squeeze(np.array(cor_inCam_point)))

    panel_1 = get_panel(cor_inCam[0], cor_inCam[1], cor_inCam[2])
    panel_2 = get_panel(cor_inCam[5], cor_inCam[6], cor_inCam[4])
    panel_3 = get_panel(cor_inCam[0], cor_inCam[3], cor_inCam[4])
    panel_4 = get_panel(cor_inCam[1], cor_inCam[2], cor_inCam[5])
    panel_5 = get_panel(cor_inCam[0], cor_inCam[1], cor_inCam[4])
    panel_6 = get_panel(cor_inCam[2], cor_inCam[3], cor_inCam[6])
    panel_list = {'panel_up':panel_1, 'panel_bot':panel_2, 'panel_front':panel_3, 'panel_behind':panel_4, 'panel_right':panel_5, 'panel_left':panel_6}

    patch_motor = []
    for point in whole_scene:
        point_cor = (point[0], point[1], point[2])
        if set_Boundingbox(panel_list, point_cor):
            patch_motor.append(point)
    return patch_motor

def Read_PCD(file_path):

    pcd = o3d.io.read_point_cloud(file_path)
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)
    return np.concatenate([points, colors], axis=-1)

def open3d_save_pcd(pc ,FileName = None):
    sampled = np.asarray(pc)
    PointCloud_koordinate = sampled[:, 0:3]

    #visuell the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
    point_cloud.colors = o3d.utility.Vector3dVector(np.float64( sampled[:, 3:]))
    o3d.io.write_point_cloud(FileName +'.pcd', point_cloud, write_ascii=True)

def points2pcd(pcd_file_path, points):

    handle = open(pcd_file_path, 'a')
    
    point_num=points.shape[0]

    handle.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')

   # int rgb = ((int)r << 16 | (int)g << 8 | (int)b); 
    for i in range(point_num):
        r,g,b = points[i,3], points[i,4], points[i,5]
        rgb = int(r)<<16 | int(g)<<8 | int(b)
        string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2]) + ' ' + str(rgb)
        handle.write(string)
    handle.close()

def transform_patch(patch_motor, cam_to_robot_transform, Cam_inBlensor_position):
    new_cor = []
    for point in patch_motor:
        # zivid to Robot
        cor_inRobot_point = transformation.camera_to_base(cam_to_robot_transform, point[0:3])    
        # Robot to Blensor real world
        cor_inReal = cor_inRobot_point
        cor_inReal[0:2] = -cor_inReal[0:2]
        # Blensor real world to Blensor camera
        cor_in_BlensorCam = get_Corordinate_inCam(Cam_inBlensor_position[0], Cam_inBlensor_position[1], Cam_inBlensor_position[2], Cam_inBlensor_position[3],
            Cam_inBlensor_position[4], Cam_inBlensor_position[5], cor_inReal)

        point_in_Blensor = np.squeeze(cor_in_BlensorCam)
        point_in_Blensor = np.hstack((point_in_Blensor, point[3:]))
        new_cor.append(np.squeeze(point_in_Blensor))
    return new_cor

def retransform_patch(patch_motor, cam_to_robot_transform, Cam_inBlensor_position):
    cor_inZivid = []
    for p in patch_motor:
        cor = get_Corordinate_inBlensor_rw(Cam_inBlensor_position[0], Cam_inBlensor_position[1], Cam_inBlensor_position[2], Cam_inBlensor_position[3],
                Cam_inBlensor_position[4], Cam_inBlensor_position[5], p[:3])
        cor[:2] = -cor[:2]

        cor_new = transformation.base_to_camera(cam_to_robot_transform, cor[0:3])
        cor_new = np.squeeze(cor_new)
        cor_new = np.hstack((cor_new[0], np.array([p[3:]])))
        cor_inZivid.append(cor_new)
    
    cor_inZivid_np = np.squeeze(np.asarray(cor_inZivid))
    return cor_inZivid_np

def find_bolts(seg_motor, eps, min_points):
    bolts = []
    for point in seg_motor:
        if point[3] == 255. and point[4] == 0. and point[5] == 0. : bolts.append(point[0:3])
    bolts = np.asarray(bolts)
    model = DBSCAN(eps=eps, min_samples=min_points)
    yhat = model.fit_predict(bolts)  # genalize label based on index
    clusters = np.unique(yhat)
    noise = []
    clusters_new = []
    positions = []
    for i in clusters:
        noise.append(i) if np.sum(i == yhat) < 200 or i == -1 else clusters_new.append(i)
    for clu in clusters_new :
        row_ix = np.where(yhat == clu)
        position = np.squeeze(np.mean(bolts[row_ix, :3], axis=1))
        positions.append(position)
    
    return positions, len(clusters_new)

def save_pcd_asIMG(pc ,FileName = None):

    sampled = np.asarray(pc)
    PointCloud_koordinate = sampled[:, 0:3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
    point_cloud.colors = o3d.utility.Vector3dVector(sampled[:, 3:])

    vis = o3d.visualization.Visualizer()  
    vis.create_window(visible=False) #works for me with False, on some systems needs to be true

    ctr = vis.get_view_control()

    vis.add_geometry(point_cloud)
    vis.get_render_option().point_size = 1.0

    ctr.rotate(0.0, -350.0)
    ctr.rotate(-500.0, 0.0)
    ctr.rotate(0.0, -500.0)
    ctr.rotate(-150.0, 0.0)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(FileName)
    vis.destroy_window()




def main():
    Corners = [(-460.03,340,240), (-741.7,340,240), (-741.7,170,240), (-460,170,240), (-460,340,5), (-741.7,340,5), (-741.7,170,5), (-460,170,5)]
    file_path = ROOT_DIR.split('data_utils')[0] + '/data'

    List_zivid = os.listdir(file_path)

    Cam_inBlensor_position = (-0.136411824,	-0.589879807, 4.067452035, 0.033196298,	0.144020323, -1.57 ) 
    transforMatrix_path = ROOT_DIR + '/meta/transformation.yaml'
    List_WholeScene = []
    for index in List_zivid:
        if os.path.splitext(index)[1] == '.pcd':
            List_WholeScene.append(index)

    cam_to_robot_transform = transformation.read_transform(transforMatrix_path)
    
    if not os.path.exists(file_path + '/Test_set'):
        os.makedirs(file_path + '/Test_set')
    k = 1
    for scene in List_WholeScene:
        pcd_path = file_path + '/' + scene
        whole_scene = Read_PCD(pcd_path)   # base open3d
       # whole_scene = whole_scene[11:, :]
       # print(whole_scene.shape)
        
        patch_motor = cut_motorPatch(Corners, cam_to_robot_transform, whole_scene)
        new_cor = transform_patch(patch_motor, cam_to_robot_transform, Cam_inBlensor_position)
        seg_motor = retransform_patch(new_cor, cam_to_robot_transform, Cam_inBlensor_position)
        patch_scene = cut_scenePatch(Corners, cam_to_robot_transform, whole_scene)
     
        break
        np_name = scene.split(".")[0]
        np.save(file_path + '/Test_set/' + 'Test_noLabeled_' + np_name, new_cor)
        print("Cutting process of zivid: %s -------------> number: %s is finished" %(scene,k))
        k += 1
        
    print(new_cor[0])

    





if __name__ == '__main__' :
    main()