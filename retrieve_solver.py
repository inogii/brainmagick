from bm import play
import torch
from omegaconf import OmegaConf
import pickle
from bm.models import SimpleConv
from bm.losses import L1Loss


def load_brennan_2019_model():
    args_simpleconv = OmegaConf.load('trained_models/brennan2019/simple_conv_params.yaml')
    more_config = OmegaConf.load('trained_models/brennan2019/simple_conv_config.yaml')
    in_channels = more_config.in_channels 
    model_chout = more_config.out_channels
    n_subjects = more_config.n_subjects
    model = SimpleConv(in_channels=in_channels, out_channels=model_chout, n_subjects=n_subjects, **args_simpleconv)

    model.load_state_dict(torch.load('trained_models/brennan2019/model_state_dict.pth'))

    print(model)

    return model

if __name__ == '__main__':

    args = ['dset.selections=[brennan2019]']
    solver = play.get_solver_from_args(args)

    solver.restore()

    print(solver.loaders)
    loaders = solver.loaders
    train_loader = loaders['train']
    valid_loader = loaders['valid']
    test_loader = loaders['test']

    # torch.save(train_loader, 'train_loader.pth')
    # torch.save(valid_loader, 'valid_loader.pth')
    # torch.save(test_loader, 'test_loader.pth')

    loaded_train_loader = torch.load('train_loader.pth')
    
    model = load_brennan_2019_model()

    batch = loaded_train_loader.dataset[0]
    output = batch.features.unsqueeze(0)
    features_mask = batch.features_mask.unsqueeze(0)
    meg = batch.meg.unsqueeze(0)
    batch.meg = meg
    inputs = dict(meg=meg)

    B, C, T = meg.shape
    print(B)
    print(C)
    print(T)

    estimate = model(inputs, batch)
    
    print(estimate)
    print(estimate.shape)

    loss = L1Loss()

    calculated_loss = loss(estimate, output, features_mask)
    print(calculated_loss)