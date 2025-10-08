import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from model_directional_query_od import Inpainting
import json

def parse_inference_args():
    """Parse command line arguments for inference"""
    parser = argparse.ArgumentParser(description='Inpainting Inference')
    
    # Model parameters - these should match your training configuration
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained model (.pth file)')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input images to inpaint')
    parser.add_argument('--output_dir', type=str, default='results/inpaint_output',
                       help='Directory to save inpainted images')
    
    # Model architecture parameters (use optimized defaults)
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[2, 4, 4, 6],
                       help='Number of blocks for each stage')
    parser.add_argument('--num_heads', type=int, nargs='+', default=[2, 2, 4, 8],
                       help='Number of attention heads for each stage')
    parser.add_argument('--channels', type=int, nargs='+', default=[24, 48, 96, 192],
                       help='Number of channels for each stage')
    parser.add_argument('--num_refinement', type=int, default=4,
                       help='Number of refinement layers')
    parser.add_argument('--expansion_factor', type=float, default=2.7582489201175653,
                       help='Expansion factor for feedforward layers')
    
    # Image processing parameters
    parser.add_argument('--image_size', type=int, default=None,
                       help='Resize images to this size (optional)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    # Config file option (alternative to individual parameters)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to JSON config file with model parameters')
    
    return parser.parse_args()

def load_config(config_path):
    """Load model configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(args, config_path):
    """Save current model configuration to JSON file"""
    config = {
        'num_blocks': args.num_blocks,
        'num_heads': args.num_heads,
        'channels': args.channels,
        'num_refinement': args.num_refinement,
        'expansion_factor': args.expansion_factor
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Model configuration saved to: {config_path}")

def create_model(num_blocks, num_heads, channels, num_refinement, expansion_factor, device):
    """Create and return the inpainting model"""
    model = Inpainting(
        num_blocks=num_blocks,
        num_heads=num_heads,
        channels=channels,
        num_refinement=num_refinement,
        expansion_factor=expansion_factor
    ).to(device)
    return model

def load_model_with_error_handling(model, model_path, device):
    """Load model state dict with comprehensive error handling"""
    try:
        # Try loading the state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle different state dict formats
        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                # If saved with additional information
                model.load_state_dict(state_dict['model_state_dict'])
                print("Loaded model from 'model_state_dict' key")
            elif 'state_dict' in state_dict:
                # Another common format
                model.load_state_dict(state_dict['state_dict'])
                print("Loaded model from 'state_dict' key")
            else:
                # Direct state dict
                model.load_state_dict(state_dict)
                print("Loaded model from direct state dict")
        else:
            raise ValueError("Invalid state dict format")
            
        print(f"Successfully loaded model from: {model_path}")
        return True
        
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"Error: Model architecture mismatch. The saved model has different dimensions.")
            print(f"This usually means the model was trained with different architecture parameters.")
            print("Please check your --num_blocks, --num_heads, --channels, --num_refinement, and --expansion_factor parameters.")
            print(f"Full error: {e}")
        else:
            print(f"Runtime error loading model: {e}")
        return False
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_image(image_path, image_size=None):
    """Load and preprocess image for inference"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Resize if specified
        if image_size is not None:
            image = image.resize((image_size, image_size), Image.LANCZOS)
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        tensor = transform(image).unsqueeze(0)
        return tensor, original_size
        
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None, None

def postprocess_output(output_tensor, original_size=None):
    """Convert model output back to PIL Image"""
    try:
        # Clamp values and convert to numpy
        output = torch.clamp(output_tensor, 0, 1)
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_np = (output_np * 255).astype('uint8')
        
        # Convert to PIL Image
        image = Image.fromarray(output_np)
        
        # Resize back to original size if needed
        if original_size is not None and image.size != original_size:
            image = image.resize(original_size, Image.LANCZOS)
        
        return image
        
    except Exception as e:
        print(f"Error postprocessing output: {e}")
        return None

def run_inference(args):
    """Main inference function"""
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load config if provided
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        args.num_blocks = config['num_blocks']
        args.num_heads = config['num_heads']
        args.channels = config['channels']
        args.num_refinement = config['num_refinement']
        args.expansion_factor = config['expansion_factor']
    
    # Print model configuration
    print("\nModel Configuration:")
    print(f"  num_blocks: {args.num_blocks}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  channels: {args.channels}")
    print(f"  num_refinement: {args.num_refinement}")
    print(f"  expansion_factor: {args.expansion_factor}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        args.num_blocks, 
        args.num_heads, 
        args.channels, 
        args.num_refinement, 
        args.expansion_factor, 
        device
    )
    
    # Calculate and print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Load model weights
    print(f"\nLoading model weights from: {args.model_path}")
    if not load_model_with_error_handling(model, args.model_path, device):
        print("Failed to load model. Exiting.")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model configuration for reference
    config_path = os.path.join(args.output_dir, 'model_config.json')
    save_config(args, config_path)
    
    # Get list of input images
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for fname in os.listdir(args.input_dir):
        if any(fname.lower().endswith(ext) for ext in image_extensions):
            image_files.append(fname)
    
    if not image_files:
        print(f"No image files found in: {args.input_dir}")
        print(f"Supported formats: {', '.join(image_extensions)}")
        return
    
    print(f"\nFound {len(image_files)} image(s) to process")
    
    # Process each image
    successful_count = 0
    for i, fname in enumerate(image_files, 1):
        print(f"\nProcessing [{i}/{len(image_files)}]: {fname}")
        
        img_path = os.path.join(args.input_dir, fname)
        
        # Preprocess image
        img_tensor, original_size = preprocess_image(img_path, args.image_size)
        if img_tensor is None:
            continue
        
        img_tensor = img_tensor.to(device)
        
        # Run inference
        try:
            with torch.no_grad():
                output = model(img_tensor)
            
            # Postprocess output
            output_image = postprocess_output(output, original_size)
            if output_image is None:
                continue
            
            # Save result
            base_name = os.path.splitext(fname)[0]
            output_name = f"{base_name}_inpainted.png"
            output_path = os.path.join(args.output_dir, output_name)
            
            output_image.save(output_path)
            print(f"  Saved: {output_path}")
            successful_count += 1
            
        except Exception as e:
            print(f"  Error processing {fname}: {e}")
            continue
    
    print(f"\nInference complete!")
    print(f"Successfully processed: {successful_count}/{len(image_files)} images")
    print(f"Results saved in: {args.output_dir}")

if __name__ == '__main__':
    args = parse_inference_args()
    print(f"Input directory: {args.input_dir}")
    run_inference(args)