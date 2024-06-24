from pathlib import Path
import yaml
import wandb
import uuid
import os

from utils import log_yaml, get_latest_checkpoint_path_from_dir
from parse import process_config_from_file
from train import TrainerSteps

def perform_training(conf_path: Path, include_path: Path, checkpoint_dir: Path | None = None, 
                     ignore_optim_state: bool = False, wandb_mode: str = "online") -> Path:
    
    wandb.init(mode=wandb_mode)
    processed_config, parsed_config = process_config_from_file(filename=str(conf_path), 
                                                               include=str(include_path),
                                                               checkpoint_path=str(get_latest_checkpoint_path_from_dir(checkpoint_dir)) if checkpoint_dir is not None else None,
                                                               ignore_optim_state=ignore_optim_state)
    log_yaml(yaml.dump(parsed_config, Dumper=yaml.Dumper))
    
    trainer = TrainerSteps(**processed_config)
    trainer.train()
    output_dir = trainer.get_output_dir()
    wandb.finish()

    return output_dir



WANDB_MODE = "offline"
conf_dir = Path(__file__).parent.parent / "conf"
conf_include_dir = conf_dir / "include"
base_model_train_conf_path =  conf_dir / "train" / "train_chebyshev_kernel_linear_regression.yml"


# 1. Load model, train it
#base_id = str(uuid.uuid4())
#print(base_id)

base_model_output_dir = perform_training(conf_path=base_model_train_conf_path, include_path=conf_include_dir, wandb_mode=WANDB_MODE)

#wandb.init(mode="offline") #
#processed_config, parsed_config = process_config_from_file(filename=str(base_model_train_conf_path), 
#                                                           include=str(conf_include_dir))

#trainer = TrainerSteps(**processed_config)
#trainer.train()
#base_model_output_dir = trainer.get_output_dir()
#base_model_weights_path = get_latest_checkpoint_path_from_dir(trainer.get_output_dir())
#log_yaml(yaml.dump(parsed_config, Dumper=yaml.Dumper))
#wandb.finish()


# 2. Save model weights, this is done through checkpoints

# 3. Load model from, and use model conf from part 1
lora_model_train_conf_path =  conf_dir / "train" / "train_cheb_lora.yml"
lora_model_output_dir = perform_training(conf_path=lora_model_train_conf_path, include_path=conf_include_dir, 
                                         checkpoint_dir=base_model_output_dir, ignore_optim_state=True, wandb_mode=WANDB_MODE)

soft_prompting_model_train_conf_path =  conf_dir / "train" / "train_cheb_soft.yml"
soft_model_output_dir = perform_training(conf_path=soft_prompting_model_train_conf_path, include_path=conf_include_dir,
                                         checkpoint_dir=base_model_output_dir, ignore_optim_state=True, wandb_mode=WANDB_MODE)

#wandb.init(mode="offline")

#processed_config, parsed_config = process_config_from_file(filename=str(lora_model_train_conf_path), 
#                                                           include=str(conf_include_dir),
#                                                           checkpoint_path=str(base_model_weights_path),
#                                                           ignore_optim_state=True)


# 4. Construct Lora model, and pass weights from part 1.
#trainer = TrainerSteps(**processed_config)
#trainer.train()
#lora_model_output_dir = trainer.get_output_dir()
#lora_model_weights_path = get_latest_checkpoint_from_dir(trainer.get_output_dir())
#log_yaml(yaml.dump(parsed_config, Dumper=yaml.Dumper))
#wandb.finish()

# 5. Save model weights from part 4
#wandb.init(mode="offline")
#soft_prompting_model_train_conf_path =  conf_dir / "train" / "train_cheb_soft.yml"
#processed_config, parsed_config = process_config_from_file(filename=str(soft_prompting_model_train_conf_path), 
#                                                           include=str(conf_include_dir),
#                                                           checkpoint_path=str(base_model_weights_path))

#rainer = TrainerSteps(**processed_config)
#trainer.train()
#log_yaml(yaml.dump(parsed_config, Dumper=yaml.Dumper))
#wandb.finish()

print("Summary:", {"base_model_output_dir": str(base_model_output_dir.absolute()), 
                   "lora_model_output_dir": str(lora_model_output_dir.absolute()),
                   "soft_prompting_model_output_dir": str(soft_model_output_dir.absolute())})