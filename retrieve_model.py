from bm import play
import torch
from bm.models.simpleconv import SimpleConv
from omegaconf import OmegaConf

if __name__ == '__main__':
    args = ["dset.selections=[brennan2019]"]
    xp = '97d170e1'
    solver = play.get_solver_from_args(args)

    #print(model.state_dict())
    solver.restore()
    model = solver.model

    print(solver.all_models)
    print(solver.datasets.train[0].meg.shape)
    model = model.to('cpu')
    
    torch.save(model.state_dict(), 'model_state_dict.pth')

    args_simpleconv = OmegaConf.load('simple_conv_params.yaml')
    more_config = OmegaConf.load('simple_conv_config.yaml')
    in_channels = more_config.in_channels
    model_chout = more_config.out_channels
    n_subjects = more_config.n_subjects
    model = SimpleConv(in_channels=in_channels, out_channels=model_chout,
                       n_subjects=n_subjects, **args_simpleconv)


    # # Load the model state dictionary on CPU
    model.load_state_dict(torch.load('model_state_dict.pth', map_location=torch.device('cpu')))

    print(model)