#!/usr/bin/env python3
"""
Checkpoint Inspector - Detailed analysis of your trained model checkpoint
"""

import torch
import json
from pathlib import Path

def inspect_checkpoint(ckpt_path):
    """Detailed inspection of checkpoint file."""
    
    print(f"🔍 Inspecting checkpoint: {ckpt_path}")
    print("=" * 80)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        print("✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return
    
    # Show all top-level keys
    print(f"\n📋 Checkpoint contents:")
    for key in checkpoint.keys():
        if key == 'model':
            print(f"  {key}: <model state dict with {len(checkpoint[key])} layers>")
        else:
            print(f"  {key}: {checkpoint[key]}")
    
    # Training metadata
    print(f"\n🏋️ Training Information:")
    print(f"  Successful rows: {checkpoint.get('successful_rows', 'N/A')}")
    print(f"  Failed rows: {checkpoint.get('failed_rows', 'N/A')}")
    print(f"  Base channels: {checkpoint.get('base', 'N/A')}")
    print(f"  Dropout: {checkpoint.get('dropout', 'N/A')}")
    
    # Disaster categories
    disaster_types = checkpoint.get('disaster_types', [])
    disaster_subtypes = checkpoint.get('disaster_subtypes', [])
    
    print(f"\n🏷️ Disaster Categories:")
    print(f"  Types ({len(disaster_types)}): {disaster_types}")
    print(f"  Subtypes ({len(disaster_subtypes)}): {disaster_subtypes}")
    
    # Training config
    if 'training_config' in checkpoint:
        print(f"\n⚙️ Training Configuration:")
        config = checkpoint['training_config']
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Analyze model architecture from state dict
    print(f"\n🧠 Model Architecture Analysis:")
    model_state = checkpoint['model']
    
    # Find input dimensions
    first_conv_key = None
    for key in model_state.keys():
        if 'conv' in key.lower() and 'weight' in key:
            first_conv_key = key
            break
    
    if first_conv_key:
        first_layer = model_state[first_conv_key]
        print(f"  First layer ({first_conv_key}): {list(first_layer.shape)}")
        print(f"  Input channels: {first_layer.shape[1]}")
        print(f"  Output channels: {first_layer.shape[0]}")
        print(f"  Kernel size: {list(first_layer.shape[2:])}")
    
    # Find output dimensions
    type_head_key = 'disaster_type_head.weight'
    subtype_head_key = 'disaster_subtype_head.weight'
    
    if type_head_key in model_state:
        type_head = model_state[type_head_key]
        print(f"  Disaster type head: {list(type_head.shape)} -> {type_head.shape[0]} classes")
    
    if subtype_head_key in model_state:
        subtype_head = model_state[subtype_head_key]
        print(f"  Disaster subtype head: {list(subtype_head.shape)} -> {subtype_head.shape[0]} classes")
    
    # Analyze all layers
    print(f"\n📊 Complete Layer Analysis:")
    conv_layers = []
    linear_layers = []
    other_layers = []
    
    for key, tensor in model_state.items():
        shape = list(tensor.shape)
        size = tensor.numel()
        
        if 'conv' in key.lower() and 'weight' in key:
            conv_layers.append((key, shape, size))
        elif 'linear' in key.lower() or any(head in key for head in ['type_head', 'subtype_head', 'classifier']):
            linear_layers.append((key, shape, size))
        else:
            other_layers.append((key, shape, size))
    
    print(f"  📐 Convolutional layers ({len(conv_layers)}):")
    for key, shape, size in conv_layers[:10]:  # Show first 10
        print(f"    {key}: {shape} ({size:,} params)")
    if len(conv_layers) > 10:
        print(f"    ... and {len(conv_layers) - 10} more conv layers")
    
    print(f"  🔗 Linear/Classification layers ({len(linear_layers)}):")
    for key, shape, size in linear_layers:
        print(f"    {key}: {shape} ({size:,} params)")
    
    # Calculate total parameters
    total_params = sum(tensor.numel() for tensor in model_state.values())
    print(f"\n📈 Model Statistics:")
    print(f"  Total layers: {len(model_state)}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")  # Assuming float32
    
    # Try to infer data configuration
    print(f"\n🔍 Data Configuration Inference:")
    if first_conv_key:
        in_channels = model_state[first_conv_key].shape[1]
        lookbacks = checkpoint.get('lookbacks', [1, 5, 10, 20])
        
        print(f"  Input channels: {in_channels}")
        print(f"  Lookback periods: {lookbacks}")
        print(f"  Estimated channels per timestep: {in_channels // len(lookbacks)}")
        
        # Try to break down channels
        channels_per_time = in_channels // len(lookbacks)
        print(f"  \n  Possible channel breakdown per timestep:")
        print(f"    S2 bands: ~10 channels")
        print(f"    DEM: 1 channel") 
        print(f"    WorldCover: 1 channel")
        print(f"    ERA5: ~5 channels")
        print(f"    SMAP: 1 channel")
        print(f"    Total estimated: ~18 channels")
        print(f"    Your model uses: {channels_per_time} channels per timestep")
    
    return checkpoint

def create_model_summary(checkpoint, output_file="model_summary.json"):
    """Create a JSON summary of the model."""
    
    model_state = checkpoint['model']
    
    summary = {
        "training_info": {
            "successful_rows": checkpoint.get('successful_rows'),
            "failed_rows": checkpoint.get('failed_rows'),
            "base_channels": checkpoint.get('base'),
            "dropout": checkpoint.get('dropout'),
        },
        "categories": {
            "disaster_types": checkpoint.get('disaster_types', []),
            "disaster_subtypes": checkpoint.get('disaster_subtypes', []),
        },
        "architecture": {},
        "training_config": checkpoint.get('training_config', {}),
    }
    
    # Architecture details
    if 'enc1.0.weight' in model_state:
        first_layer = model_state['enc1.0.weight']
        summary["architecture"]["input_channels"] = int(first_layer.shape[1])
        summary["architecture"]["first_layer_out"] = int(first_layer.shape[0])
    
    if 'disaster_type_head.weight' in model_state:
        summary["architecture"]["n_disaster_types"] = int(model_state['disaster_type_head.weight'].shape[0])
    
    if 'disaster_subtype_head.weight' in model_state:
        summary["architecture"]["n_disaster_subtypes"] = int(model_state['disaster_subtype_head.weight'].shape[0])
    
    # Layer count and params
    summary["architecture"]["total_layers"] = len(model_state)
    summary["architecture"]["total_parameters"] = sum(t.numel() for t in model_state.values())
    
    # Save summary
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Model summary saved to: {output_file}")
    return summary

def main():
    """Main inspection function."""
    
    # EDIT THIS PATH
    CKPT_PATH = "model_3d_multitask_with_negatives.ckpt.tmp_10"
    
    if not Path(CKPT_PATH).exists():
        print(f"❌ Checkpoint not found: {CKPT_PATH}")
        print("Please update CKPT_PATH in the script")
        return
    
    # Inspect checkpoint
    checkpoint = inspect_checkpoint(CKPT_PATH)
    
    if checkpoint:
        # Create summary file
        summary = create_model_summary(checkpoint)
        
        print(f"\n🎯 Key Information for Testing:")
        print(f"  Input channels: {summary['architecture'].get('input_channels', 'Unknown')}")
        print(f"  Disaster types: {summary['architecture'].get('n_disaster_types', 'Unknown')}")
        print(f"  Disaster subtypes: {summary['architecture'].get('n_disaster_subtypes', 'Unknown')}")
        print(f"  Categories: {len(summary['categories']['disaster_types'])} types, {len(summary['categories']['disaster_subtypes'])} subtypes")

if __name__ == "__main__":
    main()