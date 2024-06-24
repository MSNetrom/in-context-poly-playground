from pathlib import Path
import yaml
import wandb
import torch

from utils import log_yaml, get_latest_checkpoint_path_from_dir
from parse import process_config_from_file
from train import TrainerSteps

def perform_training(conf_path: Path, include_path: Path, checkpoint_dir: Path | None = None, 
                     ignore_optim_state: bool = False, wandb_mode: str = "online") -> Path:
    
    processed_config, parsed_config = process_config_from_file(filename=str(conf_path), 
                                                               include=str(include_path),
                                                               checkpoint_path=str(get_latest_checkpoint_path_from_dir(checkpoint_dir)) if checkpoint_dir is not None else None,
                                                               ignore_optim_state=ignore_optim_state)
    wandb.init(mode=wandb_mode, config=parsed_config)
    log_yaml(yaml.dump(parsed_config, Dumper=yaml.Dumper))
    
    trainer = TrainerSteps(**processed_config)
    trainer.train()
    output_dir = trainer.get_output_dir()
    wandb.finish()

    return output_dir

WANDB_MODE = "online"
conf_dir = Path(__file__).parent.parent / "conf"
conf_include_dir = conf_dir / "include"
base_model_train_conf_path =  conf_dir / "train" / "train_chebyshev_kernel_linear_regression.yml"

# Check cuda status
print("CUDA available:", torch.cuda.is_available())

# 1. Train base model
base_model_output_dir = perform_training(conf_path=base_model_train_conf_path, include_path=conf_include_dir, wandb_mode=WANDB_MODE)

# 2. Train Lora model
lora_model_train_conf_path =  conf_dir / "train" / "train_cheb_lora.yml"
lora_model_output_dir = perform_training(conf_path=lora_model_train_conf_path, include_path=conf_include_dir, 
                                         checkpoint_dir=base_model_output_dir, ignore_optim_state=True, wandb_mode=WANDB_MODE)

# 3. Train Soft Prompting model
soft_prompting_model_train_conf_path =  conf_dir / "train" / "train_cheb_soft.yml"
soft_model_output_dir = perform_training(conf_path=soft_prompting_model_train_conf_path, include_path=conf_include_dir,
                                         checkpoint_dir=base_model_output_dir, ignore_optim_state=True, wandb_mode=WANDB_MODE)

# 4. Print summary
print("Summary:", {"base_model_output_dir": str(base_model_output_dir.absolute()), 
                   "lora_model_output_dir": str(lora_model_output_dir.absolute()),
                   "soft_prompting_model_output_dir": str(soft_model_output_dir.absolute())})