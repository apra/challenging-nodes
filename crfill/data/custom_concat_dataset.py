from base_dataset import BaseDataset
from custom_train_dataset import CustomTrainDataset
from custom_train_negative_dataset import CustomTrainNegativeDataset


class CustomConcatDataset(BaseDataset):

    def initialize(self, opt, paths, mod):
        assert len(paths) == 2, 'Expect paths to be a list as [paths_positive, paths_negative]'
        self.paths_positive, self.paths_negative = paths

        if len(self.paths_positive) > len(self.paths_negative):
            print(f'More positive than negative, trimming positive to {len(self.paths_negative)}')
            self.paths_positive = self.paths_positive[:len(self.paths_negative)]
        elif len(self.paths_positive) < len(self.paths_negative):
            print(f'Less positive than negative, trimming negative to {len(self.paths_positive)}')
            self.paths_negative = self.paths_negative[:len(self.paths_positive)]
        else:
            print('Equal sized postive and negative')

        self.pos_dataset = CustomTrainDataset()
        self.pos_dataset.initialize(opt, self.paths_positive, mod)
        self.neg_dataset = CustomTrainNegativeDataset()
        self.neg_dataset.initialize(opt, self.paths_negative, mod)

    def __getitem__(self, index):
        pos_dict = self.pos_dataset[index]
        neg_dict = self.neg_dataset[index]
        total_dict = neg_dict  # neg_dict contains everything necessary for the gnet
        total_dict['real_image'] = pos_dict['real_image']  # this will be the image containig a tumor for the discriminator
        return total_dict

    def __len__(self):
        return len(self.pos_dataset)  # they have the same size
