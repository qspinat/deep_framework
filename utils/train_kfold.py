""" Train the model for detection task. """

import argparse

import gin

from deep_framework.trainers.kfold_trainer import KFoldTrainer


def main():
    """ Main function. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="config.gin file", type=str, required=True)
    args = parser.parse_args()
    gin.parse_config_file(args.config)
    trainer = KFoldTrainer()
    trainer.fit()


if __name__ == "__main__":
    main()
