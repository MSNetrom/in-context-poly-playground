import argparse as arg
import wandb
import os
import yaml

from typing import Any

from parse import process_config_from_file
from utils import log_yaml
#from core import TrainableModel
from train import TrainerSteps

def main(args: arg.Namespace):

    if args.checkpointfile == "":
        processed_config, parsed_config = process_config_from_file(
            filename=args.conffile,
            include=args.includedir
        )
    else:
        processed_config = process_config_from_file(
            filename=args.conffile,
            checkpoint_path=args.checkpointfile,
            include=args.includedir
        )

    init_args: dict[str, Any] = { "config" : processed_config['train'] }
    if args.projectname != "":
        init_args |= { "project" : args.projectname }
    if args.runname != "":
        init_args |= { "name" : args.runname }
    wandb.init(**init_args)

    trainer = TrainerSteps(**processed_config)

    log_yaml(yaml.dump(processed_config['train'], Dumper=yaml.Dumper))

    trainer.train()


if __name__ == "__main__":
    conf_include_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conf", "include")
    parser = arg.ArgumentParser()

    parser.add_argument("--config", '-c', type=str, default="", action='store', dest="conffile",
                        help="path to the config file to use for training")
    parser.add_argument("--resume", '-r', type=str, default="", action='store', dest="checkpointfile",
                        help="path to the checkpoint to resume training with")
    parser.add_argument("--wandb-project", type=str, default="", dest="projectname",
                        help="the project to log to in weights and biases")
    parser.add_argument("--run-name", type=str, default="", dest="runname",
                        help="what to name this run in wandb")
    parser.add_argument("--include-dir", type=str, default=conf_include_dir, action='store', dest="includedir",
                        help="from what directory to include configurations")
    args = parser.parse_args()
    main(args)
