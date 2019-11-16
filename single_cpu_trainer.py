"""
Runs a model on a single node on CPU only.
"""
import os
import logging
import numpy as np
import torch
import traceback

from test_tube import HyperOptArgumentParser, Experiment
from pytorch_lightning.models.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from model import DSANet

SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


def main(hparams):
    """
    Main training routine specific for this project
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    print('loading model...')
    model = DSANet(hparams)
    print('model built')

    # ------------------------
    # 2 INIT TEST TUBE EXP
    # ------------------------

    # init experiment
    exp = Experiment(
        name='dsanet_exp_{}_window={}_horizon={}'.format(hparams.data_name, hparams.window, hparams.horizon),
        save_dir=hparams.test_tube_save_path,
        autosave=False,
        description='test demo'
    )

    exp.argparse(hparams)
    exp.save()

    # ------------------------
    # 3 DEFINE CALLBACKS
    # ------------------------
    model_save_path = '{}/{}/{}'.format(hparams.model_save_path, exp.name, exp.version)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )

    # ------------------------
    # 4 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        experiment=exp,
        early_stop_callback=early_stop,
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)

    print('View tensorboard logs by running\ntensorboard --logdir %s' % os.getcwd())
    print('and going to http://localhost:6006 on your browser')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    # dirs
    root_dir = os.path.dirname(os.path.realpath(__file__))
    demo_log_dir = os.path.join(root_dir, 'dsanet_logs')
    checkpoint_dir = os.path.join(demo_log_dir, 'model_weights')
    test_tube_dir = os.path.join(demo_log_dir, 'test_tube_data')

    # although we user hyperOptParser, we are using it only as argparse right now
    parent_parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)

    # gpu args
    parent_parser.add_argument('--test_tube_save_path', type=str, default=test_tube_dir, help='where to save logs')
    parent_parser.add_argument('--model_save_path', type=str, default=checkpoint_dir, help='where to save model')

    # allow model to overwrite or extend args
    parser = DSANet.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    # run on HPC cluster
    print(f'RUNNING ON CPU')
    # * change the following code to comments for grid search
    main(hyperparams)


    # * recover the following code for grid search
    # hyperparams.optimize_parallel_cpu(
    #     main,
    #     nb_trials=24,    # this number needs to be adjusted according to the actual situation
    #     nb_workers=1
    # )
    