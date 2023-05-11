#!/usr/bin/python3

import argparse
import os
from trainer import Cyc_Trainer
import yaml

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/CycleGan_bi_reg_sn6.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    
    trainer = Cyc_Trainer(config)
   
    trainer.train()
    
    



###################################
if __name__ == '__main__':
    main()