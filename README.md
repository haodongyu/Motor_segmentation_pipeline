# Motor_segmentation_pipeline  

## Data Prepareration  
Please download a the example PCD file [here](https://1drv.ms/u/s!AiHw3dEQTYgAgQgVgS9Mr7MXQHMs?e=G2KyF6). Then unzip it and put the data directly at `.\data`.  
Please refer the following structure:  

Motor_segmentation_pipeline (You can also place only one PCD file here.)  
  │  
  ├─data  
  │  │  
  │  ├─ TypeA1_1.pcd  
  │  ├─ TypeA1_5.pcd  
  │  ├─ ...  
  |  └─ Scene5_1.pcd  
  └─  
  
## Run
  ```
## Check model in ./models 
## e.g., pointnet2_sem_seg
python pipeline.py --gpu 0,1 --save_img
```  
The results will exported in `.\final_output\time-based-`.  
For the training details about the pretrained model, please refer [SOTA-Networks-for-Master-Thesis-Semantic-segmentation-on-Bosch-Motors](https://github.com/haodongyu/SOTA-Networks-for-Master-Thesis-Semantic-segmentation-on-Bosch-Motors).

## Result
There will be 4 files in the default mode:  `pipeline_log.text`, `Test_TypeA1_1.txt`, `TypeA1_1.pcd`, `TypeA1_1_segResult.pcd`. If the `--save_img` mode is activated, the pipeline will produced two images, which show the orginal point cloud and the segmentated point cloud.  
- `pipeline_log.text` is used for saving the workflow's information and the position of bolts.  
- `Test_TypeA1_1.txt` is in the form of N*4. N = number of points. 4 = x, y, z-coordinate, label  
- `TypeA1_1.pcd` is orginal PCD file.
- `TypeA1_1_segResult.pcd` is the segmentation results.
