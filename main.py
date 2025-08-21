
#### to be fixed ####







#!/usr/bin/env python3
"""
Main script to orchestrate the ML pipeline
Usage: python main.py --mode [train|test|inference] --config config.yaml
"""

import argparse
import yaml
from pathlib import Path

# Import your modules
from model_builder import create_model
from dataloaders import get_train_loader, get_test_loader
from train import train_model, evaluate_model
from utils import setup_logging, save_checkpoint

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='ML Training Pipeline')
    parser.add_argument('--mode', choices=['train', 'test', 'inference'], 
                       default='train', help='Mode to run')
    parser.add_argument('--config', default='config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--checkpoint', help='Path to checkpoint file')
    parser.add_argument('--output-dir', default='outputs/', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(args.output_dir)
    
    # Create model
    model = create_model(config['model'])
    
    if args.mode == 'train':
        # Training pipeline
        train_loader = get_train_loader(config['data'])
        val_loader = get_test_loader(config['data'])
        
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config['training'],
            output_dir=args.output_dir
        )
        
        # Save final checkpoint
        save_checkpoint(trained_model, args.output_dir)
        
    elif args.mode == 'test':
        # Testing pipeline
        test_loader = get_test_loader(config['data'])
        results = evaluate_model(model, test_loader, args.checkpoint)
        print(f"Test Results: {results}")
        
    elif args.mode == 'inference':
        # Inference pipeline
        # Add your inference logic here
        pass

if __name__ == "__main__":
    main()