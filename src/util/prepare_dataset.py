from util.split import split_indices
from util.synthetic import SyntheticDataset, SyntheticSubset
from torch.utils.data import DataLoader


def prepare_dataset(config):
    '''
    Prepare the dataset for predicting one future timepoint from potentially multiple earlier timepoints.
    '''

    # Read dataset.
    dataset = SyntheticDataset(base_path=config.dataset_path,
                               image_folder=config.image_folder)
    Subset = SyntheticSubset

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = \
        split_indices(indices=indices, splits=ratios, random_seed=config.random_seed)

    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       return_format='all_subsequences')
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     return_format='all_subsequences')
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      return_format='all_subsequences')

    train_set = DataLoader(dataset=train_set,
                           batch_size=1,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=1,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel()
