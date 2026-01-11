from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_model():
    """Load the SegFormer model and processor"""
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    return processor, model

def create_colored_segmentation(pred_seg, num_classes=150):
    """Create a colored segmentation image from prediction tensor"""
    # Convert to numpy
    pred_np = pred_seg.cpu().numpy().astype(np.uint8)
    
    # Create a colormap for visualization
    colors = plt.cm.get_cmap('tab20', num_classes)
    colored_seg = colors(pred_np / num_classes)
    
    # Convert to 0-255 range and remove alpha channel
    colored_seg = (colored_seg[:, :, :3] * 255).astype(np.uint8)
    
    return colored_seg

def create_overlay_image(original_img, segmentation_array, alpha=0.5):
    """Create an overlay image with semi-transparent segmentation on original image"""
    # Resize original image to match segmentation dimensions if needed
    original_resized = original_img.resize((segmentation_array.shape[1], segmentation_array.shape[0]))
    
    # Convert original to numpy array
    original_np = np.array(original_resized)
    
    # Blend the images: result = (1-alpha) * original + alpha * segmentation
    overlay = (1 - alpha) * original_np + alpha * segmentation_array
    overlay = overlay.astype(np.uint8)
    
    return overlay

def process_single_image(image_path, processor, model, output_dir="./results"):
    """Process a single image and save results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename without extension for naming outputs
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"\nProcessing: {image_path}")
    
    try:
        # Load image
        image = Image.open(image_path)
        
        # Run inference
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Post-processing
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        
        # Get the most likely class for each pixel
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        
        # Analyze found elements
        id2label = model.config.id2label
        unique_classes = torch.unique(pred_seg).tolist()
        
        print("Elements found in this image:")
        for class_id in unique_classes:
            class_name = id2label[class_id]
            print(f"- {class_name}")
        
        # Check for building parts
        building_parts = ['wall', 'building', 'sky', 'floor', 'ceiling', 'window', 'door']
        found_parts = [id2label[c] for c in unique_classes if id2label[c] in building_parts]
        print(f"Specific Building Parts found: {found_parts}")
        
        # Create visualizations
        colored_segmentation = create_colored_segmentation(pred_seg)
        overlay_result = create_overlay_image(image, colored_segmentation, alpha=0.4)
        
        # Save individual images
        overlay_image = Image.fromarray(overlay_result)

        overlay_path = os.path.join(output_dir, f"{filename}_overlay.png")
        comparison_path = os.path.join(output_dir, f"{filename}_comparison.png")
        
        overlay_image.save(overlay_path)
        
        # Create comparison image
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        ax2.imshow(colored_segmentation)
        ax2.set_title("Segmentation Result")
        ax2.axis('off')
        
        ax3.imshow(overlay_result)
        ax3.set_title("Overlay (Original + Segmentation)")
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved files:")
        print(f"- {seg_path}")
        print(f"- {overlay_path}")
        print(f"- {comparison_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def process_multiple_images(image_paths, output_dir="./results"):
    """Process multiple images"""
    print("Loading model...")
    processor, model = load_model()
    
    successful = 0
    failed = 0
    
    for image_path in image_paths:
        if process_single_image(image_path, processor, model, output_dir):
            successful += 1
        else:
            failed += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {successful} images")
    print(f"Failed: {failed} images")
    print(f"Results saved to: {output_dir}")

# Configuration - you can modify these paths
if __name__ == "__main__":
    # Option 1: Process a single image
    single_image_path = "/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet/Barn/images/000001.jpg"
    
    # Option 2: Process multiple specific images
    image_list = [
        "/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet/Barn/images/000001.jpg",
        "/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet/Barn/images/000041.jpg",
        "/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet/Barn/images/000081.jpg",
        "/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet/Barn/images/000121.jpg",
        "/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet/Barn/images/000161.jpg",
        "/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet/Barn/images/000201.jpg",
        "/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet/Barn/images/000241.jpg",
        "/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet/Barn/images/000281.jpg",
        "/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet/Barn/images/000321.jpg",
        "/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet/Barn/images/000361.jpg",
        # Add more image paths here
    ]
    
    # Option 3: Process all images in a directory
    image_directory = "/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet/Barn/images/"
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    

    # Process multiple specific images
    process_multiple_images(image_list)
    
    # Process all images in directory
    # all_images = []
    # for ext in image_extensions:
    #     all_images.extend(glob.glob(os.path.join(image_directory, ext)))
    # process_multiple_images(all_images)
