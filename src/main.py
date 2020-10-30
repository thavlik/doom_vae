import yaml
import argparse
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning import Trainer
import torch.backends.cudnn as cudnn
import torch
import numpy as np


def basic_vae(config):
    ...


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


experiments = {
    'basic_vae': basic_vae,
}


def experiment_main(config):
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    entrypoint = config['entrypoint']
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    experiment = experiments[entrypoint](config).to(device)
    tt_logger = TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False,
    )
    runner = Trainer(default_save_path=f"{tt_logger.save_dir}",
                     min_epochs=1,
                     logger=tt_logger,
                     log_save_interval=100,
                     train_percent_check=1.,
                     val_percent_check=1.,
                     num_sanity_val_steps=5,
                     early_stop_callback=False,
                     check_val_every_n_epoch=args.log_epoch,
                     **config['trainer_params'])
    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)
    return experiment


parser = argparse.ArgumentParser(
    description='Doom VAE training entrypoint')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the experiment config file',
                    default='../configs/basic_vae.yaml')
args = parser.parse_args()
config = load_config(args.filename)
cudnn.deterministic = True
cudnn.benchmark = False
experiment_main(config)
