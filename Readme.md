##  STD_Pytorch

Due to the limited time to complete this project and my poor comprehend to the structure of such model, this code may not a correct implement of STD model, in other word, it may not work.

### Dataset：KITTI

The following are demo of ground_truth

- BEV

<img src="imgs\006397_bev.jpg" alt="006397_bev" style="zoom:50%;" />

- camera2_view(left_rgb)

<img src="imgs\006397_project.jpg" alt="006397_bev" style="zoom:50%;" />

### First Stage PGM

This stage is to generate the proposal. Because of the limitation of GPU source, I only keep up to 128 proposal in training but not the 300 in the paper.

the following is a demo of the generated proposal

<img src="imgs\005602_bev.jpg" alt="005602_bev" style="zoom:50%;" />

Due to the limited data we used for training(200 raw point clouds), the model may not perform well in the angle prediction 