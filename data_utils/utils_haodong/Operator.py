import numpy as np
import random
import open3d as o3d

def Get_ObjectID(x) :    #get all kinds of ObjectID from numpy file

    dic = []
    for i in range(x.shape[0]):
        if x[i][8] not in dic:
            dic.append(x[i][8])

    return dic


def PointCloud_Downsample(x, pointsToRemove) :   #downsample the pointcloud    #pointsToRemove = Number of points need to be deleted

    LastPointIndex = len(x) -1

    for _ in range(pointsToRemove):
        ds = np.random.uniform(0, LastPointIndex, 1)
        index = int(ds)
        x[[index, LastPointIndex]] = x[[LastPointIndex,index]]
        x = np.delete(x, LastPointIndex, 0)
        LastPointIndex -= 1
    
    return x


def ChangeLabel(x):
    if x.shape[1] == 13 :
        for i in range(x.shape[0]):
            if x[i][8] == 808464432.0 :
                x[i][8] = int(0)
            elif x[i][8] == 825307441.0 :
                x[i][8] = int(1)
            elif x[i][8] == 842150450.0 :
                x[i][8] = int(2)
            elif x[i][8] == 875836468.0 :
                x[i][8] = int(3)
            elif x[i][8] == 892679477.0 :
                x[i][8] = int(4)
            elif x[i][8] == 909522486.0 :
                x[i][8] = int(5)                              # no 3 before
    else: 
        print("The cor of numpy is not right")
            
    return x



def Resort_IDX(x):     #reset the IDX Value in the filtered numpy
    
    for i in range(x.shape[0]) :
        x[i][-1] = i

    return x


def Print_ValueOfPoint(x, NumOfPoint) :

    try:
        if x.shape[1] == 16 :
            print("Points one: timestamp {0[0]} / yaw {0[1]} / pitch {0[2]} / distance {0[3]} / distance_noise {0[4]} / Koordinate {0[5]},{0[6]},{0[7]} / Noise: {0[8]}, {0[9]}, {0[10]} / Object_id: {0[11]} / Color : {0[12]}, {0[13]}, {0[14]} / IDX: {0[15]}".format(
             x[NumOfPoint]))
        elif x.shape[1] == 13 :
            print("Points one: / distance {0[0]} / distance_noise {0[1]} / Koordinate {0[2]},{0[3]},{0[4]} / Noise: {0[5]}, {0[6]}, {0[7]} / Object_id: {0[8]} / Color : {0[9]}, {0[10]}, {0[11]} / IDX: {0[12]}".format(
                x[NumOfPoint]))
    except Exception as err :
        print(err)


def CutNumpy(x):     #drop the timestamp, yaw, pitch off and the point of (0,0,0)

    try :
        if x.shape[1] == 16 :
            x = x[:, 3:]
    except Exception as err :
        print(err)

    #Filter all points with a distance along the z coordinate small than 0
    y = x[x[:, 7] < 0]

    return y


def Read_PCD(file_path):

    pcd = o3d.io.read_point_cloud(file_path)
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)
   # print(points.shape, colors.shape)

    # pts = []
    # f = open(file_path, 'r')
    # data = f.readlines()

    # f.close()

    # get the number of points
    # line = data[9]
    # line = line.strip('\n')
    # i = line.split(' ')
    # pts_num = eval(i[-1])
    
    return np.concatenate([points, colors], axis=-1)
    # return data

    



def ChangeLabel_inPCD(file_path):

    file_data = ''
    with open(file_path, 'r') as f:
        
        for point in f.readlines() :
            point_data = point.strip('\n')
            point_data = point_data.split(' ')
            point_label = point_data[-1]

            if point_label == '808464432' :
                point = point.replace('808464432', '0')
            elif point_label == '825307441' :
                point = point.replace('825307441', '1')
            elif point_label == '842150450' :
                point = point.replace('842150450', '2')
            elif point_label == '875836468' :
                point = point.replace('875836468', '4')
            elif point_label == '892679477' :
                point = point.replace('892679477', '5')
            elif point_label == '909522486' :
                point = point.replace('909522486', '6')
            
            file_data += point
    
    with open(file_path, 'w') as f :
        f.write(file_data)

    # for point in data[11:]:
    #     point = point.strip('\n')
    #     point_data = point.split(' ')
    #     point_label = eval(point_data[-1])
    f.close()






# file_path = "F:\KIT\Masterarbeit\Dateset\Test\TestforScript\\cut00000.numpy"
# ChangeLabel_inPCD(file_path)



# scan = np.loadtxt(file_path)

# print ('11111111111111111111111111111111', scan.shape)
# print ("Points {0} / Values per point {1}".format(
#         scan.shape[0],
#         scan.shape[1]))

# np.save("Color_test", scan)

# filtered = CutNumpy(scan)
# filtered = ChangeLabel(filtered)


# print ("Points after filtered and cutted {0} / Values per point {1}".format(
#     filtered.shape[0],
#     filtered.shape[1]))

# filtered = Resort_IDX(filtered)
# np.save("cut", filtered)
# Print_ValueOfPoint(filtered, 70000)

# object_id_1 = Get_ObjectID(filtered)
# print(object_id_1)

# sampled = PointCloud_Downsample(filtered, 20000)

