import glob
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
import torchvision.transforms as transforms
from utils.misc.score_functions import score_fluoroscopy, score_time, score_xray, score_retries_cannulated_dhs, score_retries_hansson, \
                drill_dist_hansson, guidewire_dist, drill_dist_hansson, drill_dhs, stepreamer_dist, drill_dist_cannulated, guidesize_cannulated
from skimage.io import imread
import pandas as pd
import numpy as np

def flatten(l):
    return [item for sublist in l for item in sublist]

class XrayData(torch.utils.data.Dataset):
    def __init__(self, repair_type, split, source_or_target, data_path = "data/", transform = None, train_size = 0.8, test_size = 0.2, seed = 8, annotations = False):
        'Initializing data'
        #self.data_path = data_path + source_or_target + "/" #create path to data
        self.type = source_or_target
        self.split = split

        # use available device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        #set transform
        self.transform = transform
        
        # load images and scores from source

        #self.repair_type = repair_type + "/001_copenahgen_test_1"
        repair_type_source = repair_type + "/001_copenahgen_test_1"

        img_files = glob.glob(data_path + "source/" + '***/**/*.jpg', recursive=True)
        img_files = [ x for x in img_files if repair_type_source in x ]
        score_files = glob.glob(data_path + "source/" + '***/**/*.txt', recursive=True)
        score_files = [ x for x in score_files if repair_type_source in x ]
        scores = []
        scores_true = []
        maxscore = 0
                
        for file in score_files:
            with open(file) as f:
                lines = f.read()
            idx_score_end = lines.find('Score')
            tmp = lines[:idx_score_end]
            idx_score_start = tmp.rfind('\n')
            score = np.double(lines[idx_score_start+1:idx_score_end-4])

            if maxscore == 0:
                idx_maxscore_end = lines.find('Max score')
                tmp = lines[:idx_maxscore_end]
                idx_maxscore_start = tmp.rfind('\n')
                maxscore = np.double(lines[idx_maxscore_start+1:idx_maxscore_end-3])

            scores_true.append(score/maxscore)

            # find variables to be corrected in score
            if repair_type == "001_hansson_pin_system":
                var_score = []
                variables = ["Fluoroscopy (normalized)", "Total time", "Nr of X-rays", "Nr of retries", "Distal drill distance to joint surface (mm)",
                            "Guide wire distance to joint surface (mm)", "Proximal drill distance to joint surface (mm)"]
                for var in variables:
                    idx_end = lines.find(var)
                    tmp = lines[:idx_end]
                    idx_start = tmp.rfind('\n')
                    var_score.append(np.double(lines[idx_start+1:idx_end-4]))

                score += score_fluoroscopy(var_score[0])
                score += score_time(var_score[1])
                score += score_xray(var_score[2])
                score += score_retries_hansson(var_score[3])
                score += drill_dist_hansson(var_score[4])
                score += guidewire_dist(var_score[5])
                score += drill_dist_hansson(var_score[6])

                if score > maxscore:
                    score = maxscore
            
            if repair_type == "029_dynamic_hip_screw":
                var_score = []
                variables = ["Fluoroscopy (normalized)", "Time", "Nr of X-rays", "Nr of retries", "3.2 mm drill outside cortex (mm)",
                            "Guide wire distance to joint surface (mm)", "Step reamer distance to joint surface (mm)"]
                for var in variables:
                    idx_end = lines.find(var)
                    if idx_end == -1 and var == "Nr of X-rays":
                        var = 'Number of X-rays'
                        idx_end = lines.find(var)
                    elif idx_end == -1 and var == "Nr of retries":
                        var = 'Number of retries'
                        idx_end = lines.find(var)
                    tmp = lines[:idx_end]
                    idx_start = tmp.rfind('\n')
                    var_score.append(np.double(lines[idx_start+1:idx_end-4]))

                score += score_fluoroscopy(var_score[0])
                score += score_time(var_score[1])
                score += score_xray(var_score[2])
                score += score_retries_cannulated_dhs(var_score[3])
                score += drill_dhs(var_score[4])
                score += guidewire_dist(var_score[5])
                score += stepreamer_dist(var_score[6])

                if score > maxscore:
                    score = maxscore

            if repair_type == "028_cannulated_screws":
                var_score = []
                variables = ["Fluoroscopy (normalized)", "Time", "Number of X-Rays", "Nr of retries",
                                "Inferior guide wire distance to joint surface","Posterior guide wire distance to joint surface",
                                "Inferior drill distance to joint surface","Posterior drill distance to joint surface",
                                "Guide size"
                                ]
                for var in variables:
                    idx_end = lines.find(var)
                    tmp = lines[:idx_end]
                    idx_start = tmp.rfind('\n')
                    var_score.append(np.double(lines[idx_start+1:idx_end-4]))

                score += score_fluoroscopy(var_score[0])
                score += score_time(var_score[1])
                score += score_xray(var_score[2])
                score += score_retries_cannulated_dhs(var_score[3])
                score += guidewire_dist(var_score[4])
                score += guidewire_dist(var_score[5])
                score += drill_dist_cannulated(var_score[6])
                score += drill_dist_cannulated(var_score[7])
                score += guidesize_cannulated(var_score[8])

                if score > maxscore:
                    score = maxscore

                scores.append(score/maxscore)

        # create dataframe with filenames for frontal images
        df = pd.DataFrame(img_files, columns = ["image_path_frontal"])
        df = df[df.image_path_frontal.str.contains('|'.join(["frontal"]))==True]
        df["no"] = df.image_path_frontal.apply(lambda x: x[-19:-4]) # get the unique ending of the filename

        # create dataframe with scores
        df_scores = pd.DataFrame(score_files, columns = ["no"])
        df_scores["true_scores"] = scores_true
        df_scores["corrected_scores"] = scores
        df_scores.no = df_scores.no.apply(lambda x: x[-19:-4])

        # merge the paths and the scores
        # remove all admin and guest files and the images of the results. Remove all black images with score 0
        df = df.merge(df_scores, how = 'left', on = 'no')
        df = df[df.image_path_frontal.str.contains('|'.join(["admin","guest","resultTableImage"]))==False].loc[~(df["true_scores"]<=0)]

        # convert scores and image paths to lists
        scores_list = df.corrected_scores.tolist()
        scores_list = torch.Tensor(scores_list).to(self.device)
        image_paths = df.image_path_frontal.tolist()

        # divide dataset into train, val and test
        image_trainval_s, image_test_s, score_trainval_s, score_test_s = train_test_split(image_paths, scores_list, test_size=test_size, train_size=train_size, random_state=seed)
        image_train_s, image_val_s, score_train_s, score_val_s = train_test_split(image_trainval_s, score_trainval_s, test_size=0.2, train_size=0.8, random_state=seed)

        # save train and test as csv files
        image_test_df = pd.DataFrame([image_test_s,score_test_s])
        image_test_df.to_csv(data_path + 'source_' + repair_type +'_testdata.csv')
        image_val_df = pd.DataFrame([image_val_s,score_val_s])
        image_val_df.to_csv(data_path + 'source_' + repair_type +'_valdata.csv')
        image_train_df = pd.DataFrame([image_train_s,score_train_s])
        image_train_df.to_csv(data_path + 'source_' + repair_type +'_traindata.csv')

        img_files = glob.glob(data_path + "target/" + repair_type + '/*.tiff')
        img_files = [ x for x in img_files if 'ap' in x ] # only frontal image

        # divide dataset into train, val and test
        image_trainval_t, image_test_t = train_test_split(img_files, test_size=test_size, train_size=train_size, random_state=seed)
        image_train_t, image_val_t = train_test_split(image_trainval_t, test_size=0.2, train_size=0.8, random_state=seed)

        # save train and test as csv files
        image_test_df = pd.DataFrame(image_test_t)
        image_test_df.to_csv(data_path + 'target_' + repair_type +'_testdata.csv')
        image_val_df = pd.DataFrame(image_val_t)
        image_val_df.to_csv(data_path + 'target_' + repair_type +'_valdata.csv')
        image_train_df = pd.DataFrame(image_train_t)
        image_train_df.to_csv(data_path + 'target_' + repair_type +'_traindata.csv')

        # devide images into train, validation and test set
        if split == "train":
            if self.type == None:
                print("Please provide either source or target as type.")
            elif self.type == 'source':
                self.images = image_train_s
                self.scores = score_train_s
            elif self.type == 'target':
                self.images = image_train_t
        elif split == "val":
            self.images = flatten([image_val_s, image_val_t])
            score_val_t = torch.zeros(len(image_val_t))
            score_val_t[:] = float('nan')
            self.scores = flatten([score_val_s, score_val_t])
        elif split == "test":
            self.images = flatten([image_test_s, image_test_t])
            score_test_t = torch.zeros(len(image_test_t))
            score_test_t[:] = float('nan')
            self.scores = flatten([score_test_s, score_test_t])
        else:
            print("Please provide either train, val or test as split.")
    
    def __len__(self):
        'Returns the number of samples'
        return len(self.images)

    def __getitem__(self,idx):
        'Generate one sample of data'
        # get path from init
        image_path = self.images[idx]
        # read in images
        img = imread(image_path, plugin='pil')

        if self.transform:
            image = self.transform(img)

        if self.split == 'train':
            if self.type == 'source':
                # get score and assessment for the images
                score = self.scores[idx]

                return image.float(), score.float()

            elif self.type == 'target':
                return image.float()
        else:
            score = self.scores[idx]
            return image.float(), score.float()


def get_loader(repair_type, split, source_or_target = None, data_path = "data/", batch_size=16, transform = None, num_workers=0, shuffle = True, seed = 8):
    """Build and return a data loader."""
    
    dataset = XrayData(repair_type, split, source_or_target, data_path, transform, train_size = 0.8, test_size = 0.2, seed = seed)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers, drop_last = True)

    print('Finished loading dataset.')
    return data_loader