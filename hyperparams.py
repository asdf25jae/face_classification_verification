## Hyperparameters for our CNN ## 
## Jae Kang ## 
## jkang2 ## 
## github.com/asdf25jae ## 
## jkang2@andrew.cmu.edu ##


## hyperparameters for the model for training purposes ## 
epochs = 20
#learning_rate = 0.1
learning_rate = 1e-3
# try 5e-4 with gamma = 0.1 for a couple of epochs, possibly may converge better but much slower
batch_size = 64
momentum = 0.9
l2_decay = 0
weight_decay = 5e-4
num_bn_layers = 2
num_workers = 8 # number of cores used to preprocess the data in Dataloader
step_size = 5 # the number of epochs until learning rate is reduced by gamma
gamma = 0.5
milestones = [] # the milestones the learning rate is annealed by gamma at 
threshold = 0.5

## parameters of ResNet34 for building ## 

kernel_size = 3
res_block_ksize = 1
blocks_per_layer = [3,4,6,3]
stride = 1
avg_pool_stride = 4
padding = 1
final_layer_kernel_size = 4
input_in_channels = 64
num_sizes = [64, 128, 256, 512, 512]
# num_sizes = [64, 128, 256]
feat_dim = 128 # dimensions of the feature embedding 
num_classes = 2300
rgb_channels = 3

input_size = 32

## parameters of ResNet50 for building ## 

blocks_per_layer_50 = [3,4,6,3]


### parameters of MobileNetV2 ### 

mobile_std_stride = 1
mobile_kernel_size_dwise = 3
mb_feat_dim = 128 # dimensions of the mobile net's feature embedding 

