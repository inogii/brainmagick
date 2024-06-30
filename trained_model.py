import torch
from omegaconf import OmegaConf
from bm.models import SimpleConv


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
    load_brennan_2019_model()
