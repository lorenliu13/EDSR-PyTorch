from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only: # if test_only is False, it needs training data
            datasets = []
            for d in args.data_train: # look through the names in "data_train"
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG' # if dataset name contains "DIV2K-Q"  use "DIV2KJPED" as the module name, otherwise use d itself
                m = import_module('data.' + module_name.lower()) # import the module corresponding to the dataset
                datasets.append(getattr(m, module_name)(args, name=d)) # instantiate the dataset class using getattr()

            # initialize the dataloader using the concatenated dataset
            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size, # set the batch size
                shuffle=True, # if enable shuffling
                pin_memory=not args.cpu, # pin memory if args.cpu is false
                num_workers=args.n_threads, # set number of workers
            )

        # set testing dataloaders
        self.loader_test = []
        for d in args.data_test: # loop through the names in 'data_test'
            if d in ['Set5', 'Set14', 'B100', 'Urban100']: # if name in these 4 dataset
                m = import_module('data.benchmark') # import the benchmark module
                testset = getattr(m, 'Benchmark')(args, train=False, name=d) # instance the dataset class with train=False
            else: # else, intance a dataloader with batch size = 1
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
