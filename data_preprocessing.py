from torch.utils.data import DataLoader, Dataset  
import torchvision
import numpy as np 
from PIL import Image
## dataloader, dataset file ##
## this file includes all the data preprocessing and cleaning steps required for our model ## 

### loading with Dataset and DataLoader from torch.utils.data ### 
### Helper functions to make the process easier ### 



# imageFolder_dataset = torchvision.datasets.ImageFolder(root='medium/', 
#                                                       transform=torchvision.transforms.ToTensor())

# imageFolder_dataloader = DataLoader(imageFolder_dataset, batch_size=10, shuffle=True, num_workers=1)



## Below is unnecessary code from recitation, keeping it just in case it might come in handy, 
## when customization is necessary 


class ImageDataset(Dataset):
    ## image dataset for preprocessing 
    def __init__(self, file_list, target_list):
        self.file_list = file_list # complete list of all images
        self.target_list = target_list # complete list of all targets (y) 
        self.n_class = len(list(set(target_list))) # all n classes (all n targets)

    def __len__(self):
        return len(self.file_list) # length of image dataset (number of instances)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        label = self.target_list[index]
        return img, label



class ImageVerificationDataset(Dataset):
    ## image dataset for preprocessing 
    ## for verification task. Use this to preprocess verification txt files 
    # into a simpler Dataset superclass
    # data_list : np.array of (img1, img2, label)
    def __init__(self, data_list, test_mode=False):
        self.data_list = data_list # complete list of (img1 path, img2 path, label)
        # self.n_class = len(list(set(target_list))) # all n classes (all n targets)
        self.test_mode = test_mode

    def __len__(self):
        return len(self.data_list) # length of image dataset (number of pairs of instances to verify)

    def __getitem__(self, index):
        img1 = Image.open(self.data_list[index][0])
        img2 = Image.open(self.data_list[index][1][:-1])
        img1 = torchvision.transforms.ToTensor()(img1)
        img2 = torchvision.transforms.ToTensor()(img2)
        if not self.test_mode: 
            label = int(self.data_list[index][2])
            return img1, img2, label
        else: 
            return img1, img2


def parse_txt_verify(filepath, test_mode=False): 
    # function to parse txt file of format 
    # image_filepath1 image_filepath2 match 
    # where match is a binary variable indicating whether the two images are a match or not
    # output: np.ndarray of 
    f = open(filepath,'r')
    lines = f.readlines()
    result = []
    #print("result.shape :", result.shape)
    for i, line in enumerate(lines): 
        #print("line :", line) 
        line_arr = line.split(" ")
        #result[i] = line_arr
        if test_mode:
            img1, img2 = line_arr 
            img1 = "hw2p2_check/test_verification/" + img1
            img2 = "hw2p2_check/test_verification/" + img2
            result.append([img1, img2])
        if not test_mode: 
            img1, img2, label = line_arr 
            img1 = "hw2p2_check/validation_verification/" + img1 
            img2 = "hw2p2_check/validation_verification/" + img2
            result.append([img1, img2, label])
        # img_1, img_2, label = line_arr[0], line_arr[1], line_arr[2]
    return result  

"""
def parse_data(datadir):
    img_list = []
    ID_list = []
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
                ID_list.append(root.split('/')[-1])
                # print ("imglist: ",img_list,"\n Id list: ",ID_list)

    # construct a dictionary, where key and value correspond to ID and target
    uniqueID_list = list(set(ID_list))
    class_n = len(uniqueID_list)
    target_dict = dict(zip(uniqueID_list, range(class_n)))
    label_list = [target_dict[ID_key] for ID_key in ID_list]
    print(label_list[:50])
    print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
    return img_list, label_list, class_n
"""

#img_list, label_list, class_n = parse_data('../train_data/medium')

#trainset = ImageDataset(img_list, label_list)

#train_data_item, train_data_label = trainset.__getitem__(0)

#print('data item shape: {}\t data item label: {}'.format(train_data_item.shape, train_data_label))

#dataloader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1, drop_last=False)





"""
def load_raw(path, name):
    if name == "test": 
        return np.load(os.path.join(path, '{}.npy'.format(name)), encoding='bytes')
    else: 
        return (
        np.load(os.path.join(path, '{}.npy'.format(name)), encoding='bytes'), 
        np.load(os.path.join(path, '{}_labels.npy'.format(name)), encoding='bytes')
    )


def get_dataset(file, list_file, context_len, gpu, isTestSet=False):
    return UtterancesDataset(file, list_file, context_len, gpu, isTestSet)


def get_dataloader(x, list_file, context_len, gpu, isTestSet = False, shuffle = True, num_workers=0, isTrainSet=False):
    dataset = get_dataset(x, list_file, context_len, gpu, isTestSet)
    if isTrainSet: 
        # trainSet set 
        return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
    elif (not isTrainSet and not isTestSet): 
        # validation set 
        return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    else: 
        # test set 
        return DataLoader( 
            dataset, batch_size=batch_size, shuffle=False
        )

"""

## other dead code, in storage ## 


"""
class NetModule():
        # simple wrapper module for Neural Net 
        # that will help maintain all the neural net's saved parameters 
        # and use that later for inference, generalization etc. 

        def __init__(self, net, optimizer, criterion, train_data_loader, test_dataloader):
			self.net = net  
			self.optimizer = optimizer 
			self.criterion = criterion
			self.train_dataloader = train_dataloader 
			self.test_dataloader = test_dataloader
"""
