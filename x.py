import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import glob
from tqdm import tqdm
import colorsys
import argparse

def generate_inpainting_mask(height=256, width=256, mask_type="irregular", coverage=None):
    """
    Generate realistic inpainting masks that remove image regions
    
    Args:
        height (int): Height of the mask
        width (int): Width of the mask  
        mask_type (str): "irregular", "rectangular", "circular", "stroke"
        coverage (float): Percentage of image to mask (0.1 to 0.4)
        
    Returns:
        numpy.ndarray: Binary mask (0=keep, 255=remove/inpaint)
    """
    if coverage is None:
        coverage = random.uniform(0.15, 0.35)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    target_area = int(height * width * coverage)
    current_area = 0
    
    if mask_type == "irregular":
        mask = _generate_irregular_mask(height, width, coverage)
    elif mask_type == "rectangular":
        mask = _generate_rectangular_mask(height, width, coverage)
    elif mask_type == "circular":
        mask = _generate_circular_mask(height, width, coverage)
    elif mask_type == "stroke":
        mask = _generate_stroke_mask(height, width, coverage)
    else:
        # Mixed approach
        mask = _generate_mixed_mask(height, width, coverage)
    
    return mask

def _generate_irregular_mask(height, width, coverage):
    """Generate irregular blob-like masks"""
    mask = np.zeros((height, width), dtype=np.uint8)
    target_area = int(height * width * coverage)
    
    # Generate 2-5 irregular blobs
    num_blobs = random.randint(2, 5)
    
    for _ in range(num_blobs):
        # Random center
        center_x = random.randint(width//4, 3*width//4)
        center_y = random.randint(height//4, 3*height//4)
        
        # Create irregular shape using multiple overlapping circles
        blob_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Main blob
        main_radius = random.randint(min(height, width)//8, min(height, width)//4)
        cv2.circle(blob_mask, (center_x, center_y), main_radius, 255, -1)
        
        # Add 3-8 smaller overlapping circles for irregular shape
        num_sub_circles = random.randint(3, 8)
        for _ in range(num_sub_circles):
            offset_x = random.randint(-main_radius, main_radius)
            offset_y = random.randint(-main_radius, main_radius)
            sub_radius = random.randint(main_radius//3, main_radius//2)
            
            sub_center_x = np.clip(center_x + offset_x, 0, width-1)
            sub_center_y = np.clip(center_y + offset_y, 0, height-1)
            
            cv2.circle(blob_mask, (sub_center_x, sub_center_y), sub_radius, 255, -1)
        
        # Smooth the blob edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blob_mask = cv2.morphologyEx(blob_mask, cv2.MORPH_CLOSE, kernel)
        blob_mask = cv2.GaussianBlur(blob_mask, (7, 7), 0)
        _, blob_mask = cv2.threshold(blob_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Add to main mask
        mask = cv2.bitwise_or(mask, blob_mask)
        
        # Check if we've reached target coverage
        if np.sum(mask > 0) >= target_area:
            break
    
    return mask

def _generate_rectangular_mask(height, width, coverage):
    """Generate rectangular masks"""
    mask = np.zeros((height, width), dtype=np.uint8)
    target_area = int(height * width * coverage)
    
    # Generate 1-3 rectangles
    num_rects = random.randint(1, 3)
    
    for _ in range(num_rects):
        # Random rectangle dimensions
        rect_w = random.randint(width//6, width//2)
        rect_h = random.randint(height//6, height//2)
        
        # Random position
        x = random.randint(0, max(1, width - rect_w))
        y = random.randint(0, max(1, height - rect_h))
        
        cv2.rectangle(mask, (x, y), (x + rect_w, y + rect_h), 255, -1)
        
        if np.sum(mask > 0) >= target_area:
            break
    
    return mask

def _generate_circular_mask(height, width, coverage):
    """Generate circular masks"""
    mask = np.zeros((height, width), dtype=np.uint8)
    target_area = int(height * width * coverage)
    
    # Generate 1-4 circles
    num_circles = random.randint(1, 4)
    
    for _ in range(num_circles):
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        radius = random.randint(min(height, width)//8, min(height, width)//3)
        
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        if np.sum(mask > 0) >= target_area:
            break
    
    return mask

def _generate_stroke_mask(height, width, coverage):
    """Generate stroke-like masks"""
    mask = np.zeros((height, width), dtype=np.uint8)
    target_area = int(height * width * coverage)
    
    # Generate 5-15 strokes
    num_strokes = random.randint(5, 15)
    
    for _ in range(num_strokes):
        # Random start and end points
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        
        # Random thickness
        thickness = random.randint(3, 15)
        
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
        
        if np.sum(mask > 0) >= target_area:
            break
    
    # Smooth strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def _generate_mixed_mask(height, width, coverage):
    """Generate mixed mask types"""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Randomly combine different mask types
    mask_types = ["irregular", "rectangular", "circular", "stroke"]
    selected_types = random.sample(mask_types, random.randint(1, 3))
    
    coverage_per_type = coverage / len(selected_types)
    
    for mask_type in selected_types:
        if mask_type == "irregular":
            temp_mask = _generate_irregular_mask(height, width, coverage_per_type)
        elif mask_type == "rectangular":
            temp_mask = _generate_rectangular_mask(height, width, coverage_per_type)
        elif mask_type == "circular":
            temp_mask = _generate_circular_mask(height, width, coverage_per_type)
        else:  # stroke
            temp_mask = _generate_stroke_mask(height, width, coverage_per_type)
        
        mask = cv2.bitwise_or(mask, temp_mask)
    
    return mask

def apply_inpainting_mask(image, mask, fill_method="white"):
    """
    Apply inpainting mask to image
    
    Args:
        image (numpy.ndarray): Original image
        mask (numpy.ndarray): Binary mask
        fill_method (str): "white", "noise", "blur"
        
    Returns:
        numpy.ndarray: Masked image ready for inpainting
    """
    result = image.copy()
    
    if fill_method == "white":
        # Fill masked areas with white
        result[mask > 0] = 255
    elif fill_method == "noise":
        # Fill with random noise
        noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)
        result[mask > 0] = noise[mask > 0]
    elif fill_method == "blur":
        # Fill with heavily blurred version
        blurred = cv2.GaussianBlur(image, (51, 51), 0)
        result[mask > 0] = blurred[mask > 0]
    
    return result

def generate_face_aware_mask(height=256, width=256, coverage=None):
    """
    Generate masks that avoid critical face regions
    """
    if coverage is None:
        coverage = random.uniform(0.15, 0.3)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define face region (approximate)
    face_center_x, face_center_y = width // 2, height // 2
    face_radius = min(width, height) // 3
    
    # Generate mask avoiding eye and mouth regions
    eye_y = int(face_center_y - face_radius * 0.2)
    mouth_y = int(face_center_y + face_radius * 0.3)
    
    # Create exclusion zones
    exclusion_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Eyes exclusion
    cv2.circle(exclusion_mask, (int(face_center_x - face_radius * 0.3), eye_y), 
               face_radius // 6, 255, -1)
    cv2.circle(exclusion_mask, (int(face_center_x + face_radius * 0.3), eye_y), 
               face_radius // 6, 255, -1)
    
    # Mouth exclusion
    cv2.ellipse(exclusion_mask, (face_center_x, mouth_y), 
                (face_radius // 4, face_radius // 8), 0, 0, 360, 255, -1)
    
    # Generate base mask
    base_mask = _generate_irregular_mask(height, width, coverage * 1.5)
    
    # Remove exclusion zones
    mask = cv2.bitwise_and(base_mask, cv2.bitwise_not(exclusion_mask))
    
    return mask

# ENHANCED COMPOSITE MASK FUNCTIONS
def generate_enhanced_artistic_mask(height=256, width=256, style="high_coverage_splashes"):
    """
    Generate enhanced artistic composite masks with higher opacity and complexity
    
    Args:
        height (int): Height of the mask
        width (int): Width of the mask
        style (str): Enhanced artistic styles
        
    Returns:
        tuple: (composite_mask, color_overlay) - mask and colored overlay
    """
    if style == "high_coverage_splashes":
        return _generate_high_coverage_splashes(height, width)
    elif style == "complex_geometric":
        return _generate_complex_geometric(height, width)
    elif style == "dense_paint_strokes":
        return _generate_dense_paint_strokes(height, width)
    elif style == "vibrant_organic":
        return _generate_vibrant_organic(height, width)
    elif style == "layered_artistic":
        return _generate_layered_artistic(height, width)
    else:
        return _generate_enhanced_mixed(height, width)

def _generate_high_coverage_splashes(height, width):
    """Generate high-coverage colorful splashes with complex patterns"""
    mask = np.zeros((height, width), dtype=np.uint8)
    color_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Generate 15-25 colorful splashes for higher coverage
    num_splashes = random.randint(15, 25)
    
    for i in range(num_splashes):
        # More distributed positioning
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        
        # Larger, more varied sizes
        radius = random.randint(min(height, width)//10, min(height, width)//4)
        
        # Generate ultra-vibrant colors
        hue = random.uniform(0, 1)
        saturation = random.uniform(0.8, 1.0)
        value = random.uniform(0.8, 1.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
        
        # Create complex splash with multiple layers
        splash_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Main splash with higher opacity
        cv2.circle(splash_mask, (center_x, center_y), radius, 255, -1)
        
        # Add more complex irregular edges
        num_edges = random.randint(8, 15)
        for _ in range(num_edges):
            angle = random.uniform(0, 2 * np.pi)
            distance = random.randint(radius//3, int(radius * 1.2))
            edge_x = int(center_x + distance * np.cos(angle))
            edge_y = int(center_y + distance * np.sin(angle))
            edge_radius = random.randint(radius//3, radius//2)
            
            if 0 <= edge_x < width and 0 <= edge_y < height:
                cv2.circle(splash_mask, (edge_x, edge_y), edge_radius, 255, -1)
        
        # Add fractal-like details
        for _ in range(random.randint(3, 8)):
            angle = random.uniform(0, 2 * np.pi)
            distance = random.randint(radius//2, radius)
            detail_x = int(center_x + distance * np.cos(angle))
            detail_y = int(center_y + distance * np.sin(angle))
            detail_radius = random.randint(radius//6, radius//4)
            
            if 0 <= detail_x < width and 0 <= detail_y < height:
                cv2.circle(splash_mask, (detail_x, detail_y), detail_radius, 180, -1)
        
        # Enhanced smoothing for organic look
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        splash_mask = cv2.morphologyEx(splash_mask, cv2.MORPH_CLOSE, kernel)
        splash_mask = cv2.GaussianBlur(splash_mask, (11, 11), 0)
        _, splash_mask = cv2.threshold(splash_mask, 80, 255, cv2.THRESH_BINARY)
        
        # Add to composite
        mask = cv2.bitwise_or(mask, splash_mask)
        color_overlay[splash_mask > 0] = color
    
    return mask, color_overlay

def _generate_complex_geometric(height, width):
    """Generate complex geometric patterns with layered shapes"""
    mask = np.zeros((height, width), dtype=np.uint8)
    color_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Generate more geometric shapes for complexity
    num_shapes = random.randint(12, 20)
    
    for i in range(num_shapes):
        shape_type = random.choice(['circle', 'rectangle', 'triangle', 'polygon', 'ellipse'])
        
        # Generate highly saturated colors
        hue = random.uniform(0, 1)
        saturation = random.uniform(0.8, 1.0)
        value = random.uniform(0.7, 1.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
        
        shape_mask = np.zeros((height, width), dtype=np.uint8)
        
        if shape_type == 'circle':
            center = (random.randint(0, width), random.randint(0, height))
            radius = random.randint(25, 80)
            cv2.circle(shape_mask, center, radius, 255, -1)
            
        elif shape_type == 'rectangle':
            x1 = random.randint(0, width-50)
            y1 = random.randint(0, height-50)
            x2 = x1 + random.randint(40, 120)
            y2 = y1 + random.randint(40, 120)
            cv2.rectangle(shape_mask, (x1, y1), (min(x2, width), min(y2, height)), 255, -1)
            
        elif shape_type == 'triangle':
            pts = np.array([
                [random.randint(0, width), random.randint(0, height)],
                [random.randint(0, width), random.randint(0, height)],
                [random.randint(0, width), random.randint(0, height)]
            ], np.int32)
            cv2.fillPoly(shape_mask, [pts], 255)
            
        elif shape_type == 'ellipse':
            center = (random.randint(0, width), random.randint(0, height))
            axes = (random.randint(30, 80), random.randint(20, 60))
            angle = random.randint(0, 180)
            cv2.ellipse(shape_mask, center, axes, angle, 0, 360, 255, -1)
            
        else:  # polygon
            num_points = random.randint(5, 8)
            center_x, center_y = random.randint(50, width-50), random.randint(50, height-50)
            radius = random.randint(30, 70)
            pts = []
            for j in range(num_points):
                angle = (2 * np.pi * j) / num_points
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                pts.append([x, y])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(shape_mask, [pts], 255)
        
        # Add rotation and transformation effects
        if random.random() > 0.5:
            angle = random.randint(-45, 45)
            center = (width//2, height//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            shape_mask = cv2.warpAffine(shape_mask, M, (width, height))
        
        # Reduce harsh edges but maintain opacity
        shape_mask = cv2.GaussianBlur(shape_mask, (3, 3), 0)
        
        mask = cv2.bitwise_or(mask, shape_mask)
        color_overlay[shape_mask > 127] = color
    
    return mask, color_overlay

def _generate_dense_paint_strokes(height, width):
    """Generate dense paint stroke patterns with high coverage"""
    mask = np.zeros((height, width), dtype=np.uint8)
    color_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Generate many more strokes for dense coverage
    num_strokes = random.randint(25, 40)
    
    for i in range(num_strokes):
        # Create longer, more varied strokes
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        length = random.randint(50, 150)
        angle = random.uniform(0, 2 * np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        x2 = np.clip(x2, 0, width-1)
        y2 = np.clip(y2, 0, height-1)
        
        thickness = random.randint(12, 30)
        
        # Generate vibrant colors
        hue = random.uniform(0, 1)
        saturation = random.uniform(0.7, 1.0)
        value = random.uniform(0.8, 1.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
        
        stroke_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.line(stroke_mask, (x1, y1), (x2, y2), 255, thickness)
        
        # Add brush texture and organic feel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        stroke_mask = cv2.morphologyEx(stroke_mask, cv2.MORPH_CLOSE, kernel)
        stroke_mask = cv2.GaussianBlur(stroke_mask, (5, 5), 0)
        
        mask = cv2.bitwise_or(mask, stroke_mask)
        color_overlay[stroke_mask > 100] = color
    
    return mask, color_overlay

def _generate_vibrant_organic(height, width):
    """Generate organic, flowing patterns with vibrant colors"""
    mask = np.zeros((height, width), dtype=np.uint8)
    color_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create flowing organic shapes
    num_flows = random.randint(8, 15)
    
    for i in range(num_flows):
        # Create flowing path
        start_x = random.randint(0, width)
        start_y = random.randint(0, height)
        
        flow_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Generate vibrant color
        hue = random.uniform(0, 1)
        saturation = random.uniform(0.8, 1.0)
        value = random.uniform(0.8, 1.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
        
        # Create organic flow with multiple connected circles
        current_x, current_y = start_x, start_y
        num_segments = random.randint(15, 25)
        
        for j in range(num_segments):
            radius = random.randint(15, 35)
            cv2.circle(flow_mask, (int(current_x), int(current_y)), radius, 255, -1)
            
            # Move in organic pattern
            angle = random.uniform(0, 2 * np.pi)
            step = random.randint(10, 25)
            current_x += step * np.cos(angle)
            current_y += step * np.sin(angle)
            
            # Keep within bounds
            current_x = np.clip(current_x, radius, width - radius)
            current_y = np.clip(current_y, radius, height - radius)
        
        # Smooth for organic look
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        flow_mask = cv2.morphologyEx(flow_mask, cv2.MORPH_CLOSE, kernel)
        flow_mask = cv2.GaussianBlur(flow_mask, (13, 13), 0)
        _, flow_mask = cv2.threshold(flow_mask, 100, 255, cv2.THRESH_BINARY)
        
        mask = cv2.bitwise_or(mask, flow_mask)
        color_overlay[flow_mask > 0] = color
    
    return mask, color_overlay

def _generate_layered_artistic(height, width):
    """Generate complex layered artistic effects"""
    mask = np.zeros((height, width), dtype=np.uint8)
    color_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Layer 1: Base splashes
    base_mask, base_overlay = _generate_high_coverage_splashes(height, width)
    mask = cv2.bitwise_or(mask, base_mask)
    color_overlay[base_mask > 0] = base_overlay[base_mask > 0]
    
    # Layer 2: Geometric accents
    geo_mask, geo_overlay = _generate_complex_geometric(height, width)
    # Reduce geometric coverage to 60% for layering
    geo_mask = (geo_mask * 0.6).astype(np.uint8)
    mask = cv2.bitwise_or(mask, geo_mask)
    blend_areas = geo_mask > 0
    color_overlay[blend_areas] = geo_overlay[blend_areas]
    
    # Layer 3: Fine stroke details
    stroke_mask, stroke_overlay = _generate_dense_paint_strokes(height, width)
    # Reduce stroke coverage to 40% for fine details
    stroke_mask = (stroke_mask * 0.4).astype(np.uint8)
    mask = cv2.bitwise_or(mask, stroke_mask)
    detail_areas = stroke_mask > 0
    color_overlay[detail_areas] = stroke_overlay[detail_areas]
    
    return mask, color_overlay

def _generate_enhanced_mixed(height, width):
    """Generate enhanced mixed artistic effects with higher complexity"""
    styles = ['high_coverage_splashes', 'complex_geometric', 'dense_paint_strokes', 'vibrant_organic']
    selected_styles = random.sample(styles, random.randint(2, 3))
    
    final_mask = np.zeros((height, width), dtype=np.uint8)
    final_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i, style in enumerate(selected_styles):
        if style == 'high_coverage_splashes':
            mask, overlay = _generate_high_coverage_splashes(height, width)
        elif style == 'complex_geometric':
            mask, overlay = _generate_complex_geometric(height, width)
        elif style == 'dense_paint_strokes':
            mask, overlay = _generate_dense_paint_strokes(height, width)
        else:  # vibrant_organic
            mask, overlay = _generate_vibrant_organic(height, width)
        
        # Blend with existing (priority to later layers)
        final_mask = cv2.bitwise_or(final_mask, mask)
        blend_areas = mask > 0
        final_overlay[blend_areas] = overlay[blend_areas]
    
    return final_mask, final_overlay

def apply_enhanced_composite_mask(image, mask, color_overlay, opacity=0.85, preserve_face=True):
    """
    Apply enhanced composite mask with higher opacity and face preservation
    
    Args:
        image (numpy.ndarray): Original image (BGR)
        mask (numpy.ndarray): Binary mask
        color_overlay (numpy.ndarray): Color overlay (BGR)
        opacity (float): High opacity for strong effect (0.7 to 1.0)
        preserve_face (bool): Whether to preserve facial features
        
    Returns:
        numpy.ndarray: Image with enhanced composite mask applied
    """
    result = image.copy().astype(np.float32)
    overlay = color_overlay.astype(np.float32)
    
    # Create face preservation mask if requested
    if preserve_face:
        height, width = image.shape[:2]
        face_center_x, face_center_y = width // 2, height // 2
        face_radius = min(width, height) // 4
        
        preservation_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Preserve eye regions
        eye_y = int(face_center_y - face_radius * 0.2)
        cv2.circle(preservation_mask, (int(face_center_x - face_radius * 0.3), eye_y), 
                   face_radius // 5, 255, -1)
        cv2.circle(preservation_mask, (int(face_center_x + face_radius * 0.3), eye_y), 
                   face_radius // 5, 255, -1)
        
        # Preserve mouth region
        mouth_y = int(face_center_y + face_radius * 0.3)
        cv2.ellipse(preservation_mask, (face_center_x, mouth_y), 
                    (face_radius // 3, face_radius // 6), 0, 0, 360, 255, -1)
        
        # Reduce mask intensity in preservation areas
        mask_float = mask.astype(np.float32) / 255.0
        preservation_float = preservation_mask.astype(np.float32) / 255.0
        mask_float = mask_float * (1 - preservation_float * 0.7)  # Reduce by 70% in face areas
        mask = (mask_float * 255).astype(np.uint8)
    
    # Enhanced overlay blend with multiple blend modes
    mask_norm = (mask > 0).astype(np.float32)
    
    # Color dodge effect for vibrant colors
    for c in range(3):
        base = result[:, :, c] / 255.0
        blend = overlay[:, :, c] / 255.0
        
        # Enhanced color dodge with saturation boost
        dodge_result = np.where(
            blend < 1.0,
            np.minimum(1.0, base / (1.001 - blend)),  # Avoid division by zero
            1.0
        )
        
        # Screen blend for additional brightness
        screen_result = 1 - ((1 - base) * (1 - blend))
        
        # Combine dodge and screen
        combined = (dodge_result * 0.6 + screen_result * 0.4)
        result[:, :, c] = combined * 255
    
    # Apply high opacity blending
    mask_3d = np.stack([mask_norm] * 3, axis=2)
    final_result = image.astype(np.float32) * (1 - mask_3d * opacity) + result * mask_3d * opacity
    
    # Convert to HSV for saturation enhancement
    final_result_hsv = cv2.cvtColor(final_result.astype(np.uint8), cv2.COLOR_BGR2HSV)
    final_result_hsv[:, :, 1] = np.clip(final_result_hsv[:, :, 1] * 1.2, 0, 255)  # Boost saturation
    final_result = cv2.cvtColor(final_result_hsv, cv2.COLOR_HSV2BGR)
    
    return np.clip(final_result, 0, 255).astype(np.uint8)

def generate_artistic_composite_mask(height=256, width=256, style="colorful_splashes"):
    """
    Generate artistic composite masks with colored overlays
    
    Args:
        height (int): Height of the mask
        width (int): Width of the mask
        style (str): Artistic style - "colorful_splashes", "geometric_overlay", "paint_strokes", "mixed_artistic"
        
    Returns:
        tuple: (composite_mask, color_overlay) - binary mask and colored overlay
    """
    if style == "colorful_splashes":
        return _generate_colorful_splashes(height, width)
    elif style == "geometric_overlay":
        return _generate_geometric_overlay(height, width)
    elif style == "paint_strokes":
        return _generate_paint_strokes(height, width)
    elif style == "mixed_artistic":
        return _generate_mixed_artistic(height, width)
    else:
        return _generate_colorful_splashes(height, width)

def _generate_colorful_splashes(height, width):
    """Generate colorful paint splash effects"""
    mask = np.zeros((height, width), dtype=np.uint8)
    color_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Generate 8-15 colorful splashes
    num_splashes = random.randint(8, 15)
    
    for i in range(num_splashes):
        # Random center position
        center_x = random.randint(width//6, 5*width//6)
        center_y = random.randint(height//6, 5*height//6)
        
        # Vary splash sizes
        radius = random.randint(min(height, width)//12, min(height, width)//5)
        
        # Generate vibrant colors
        hue = random.uniform(0, 1)
        saturation = random.uniform(0.7, 1.0)
        value = random.uniform(0.8, 1.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))  # BGR format
        
        # Create splash with irregular edges
        splash_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Main circular splash
        cv2.circle(splash_mask, (center_x, center_y), radius, 255, -1)
        
        # Add irregular edges with smaller circles
        num_edges = random.randint(5, 10)
        for _ in range(num_edges):
            angle = random.uniform(0, 2 * np.pi)
            distance = random.randint(radius//2, int(radius * 1.3))
            edge_x = int(center_x + distance * np.cos(angle))
            edge_y = int(center_y + distance * np.sin(angle))
            edge_radius = random.randint(radius//4, radius//2)
            
            if 0 <= edge_x < width and 0 <= edge_y < height:
                cv2.circle(splash_mask, (edge_x, edge_y), edge_radius, 255, -1)
        
        # Smooth edges for organic look
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        splash_mask = cv2.morphologyEx(splash_mask, cv2.MORPH_CLOSE, kernel)
        splash_mask = cv2.GaussianBlur(splash_mask, (9, 9), 0)
        _, splash_mask = cv2.threshold(splash_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Add to composite
        mask = cv2.bitwise_or(mask, splash_mask)
        color_overlay[splash_mask > 0] = color
    
    return mask, color_overlay

def _generate_geometric_overlay(height, width):
    """Generate geometric pattern overlays"""
    mask = np.zeros((height, width), dtype=np.uint8)
    color_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Generate 6-12 geometric shapes
    num_shapes = random.randint(6, 12)
    
    for i in range(num_shapes):
        shape_type = random.choice(['circle', 'rectangle', 'triangle', 'polygon'])
        
        # Generate bright colors
        hue = random.uniform(0, 1)
        saturation = random.uniform(0.6, 1.0)
        value = random.uniform(0.7, 1.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
        
        shape_mask = np.zeros((height, width), dtype=np.uint8)
        
        if shape_type == 'circle':
            center = (random.randint(0, width), random.randint(0, height))
            radius = random.randint(20, 60)
            cv2.circle(shape_mask, center, radius, 255, -1)
            
        elif shape_type == 'rectangle':
            x1 = random.randint(0, width-30)
            y1 = random.randint(0, height-30)
            x2 = x1 + random.randint(30, 100)
            y2 = y1 + random.randint(30, 100)
            cv2.rectangle(shape_mask, (x1, y1), (min(x2, width), min(y2, height)), 255, -1)
            
        elif shape_type == 'triangle':
            pts = np.array([
                [random.randint(0, width), random.randint(0, height)],
                [random.randint(0, width), random.randint(0, height)],
                [random.randint(0, width), random.randint(0, height)]
            ], np.int32)
            cv2.fillPoly(shape_mask, [pts], 255)
            
        else:  # polygon
            num_points = random.randint(5, 8)
            center_x, center_y = random.randint(40, width-40), random.randint(40, height-40)
            radius = random.randint(25, 60)
            pts = []
            for j in range(num_points):
                angle = (2 * np.pi * j) / num_points
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                pts.append([x, y])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(shape_mask, [pts], 255)
        
        # Slight blur for softer edges
        shape_mask = cv2.GaussianBlur(shape_mask, (3, 3), 0)
        
        mask = cv2.bitwise_or(mask, shape_mask)
        color_overlay[shape_mask > 127] = color
    
    return mask, color_overlay

def _generate_paint_strokes(height, width):
    """Generate paint stroke patterns"""
    mask = np.zeros((height, width), dtype=np.uint8)
    color_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Generate 10-20 paint strokes
    num_strokes = random.randint(10, 20)
    
    for i in range(num_strokes):
        # Random stroke endpoints
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        length = random.randint(40, 120)
        angle = random.uniform(0, 2 * np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        x2 = np.clip(x2, 0, width-1)
        y2 = np.clip(y2, 0, height-1)
        
        thickness = random.randint(8, 20)
        
        # Generate stroke colors
        hue = random.uniform(0, 1)
        saturation = random.uniform(0.6, 1.0)
        value = random.uniform(0.7, 1.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
        
        stroke_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.line(stroke_mask, (x1, y1), (x2, y2), 255, thickness)
        
        # Add brush texture
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        stroke_mask = cv2.morphologyEx(stroke_mask, cv2.MORPH_CLOSE, kernel)
        stroke_mask = cv2.GaussianBlur(stroke_mask, (3, 3), 0)
        
        mask = cv2.bitwise_or(mask, stroke_mask)
        color_overlay[stroke_mask > 100] = color
    
    return mask, color_overlay

def create_enhanced_composite_artistic_image(image_path, output_path, style="high_coverage_splashes", 
                                           opacity=0.85, preserve_face=True):
    """
    Create enhanced composite image with artistic masks
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save output
        style (str): Enhanced artistic style
        opacity (float): Blending opacity (0.7 to 1.0)
        preserve_face (bool): Whether to preserve facial features
        
    Returns:
        bool: Success status
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return False
        
        # Resize if needed
        original_size = image.shape[:2]
        if original_size != (256, 256):
            image = cv2.resize(image, (256, 256))
        
        # Generate enhanced artistic composite mask
        mask, color_overlay = generate_enhanced_artistic_mask(256, 256, style)
        
        # Apply enhanced composite mask
        result = apply_enhanced_composite_mask(image, mask, color_overlay, opacity, preserve_face)
        
        # Resize back to original if needed
        if original_size != (256, 256):
            result = cv2.resize(result, (original_size[1], original_size[0]))
        
        # Save result
        cv2.imwrite(output_path, result)
        print(f"Enhanced composite artistic image saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating enhanced composite image: {e}")
        return False

def batch_create_enhanced_composite_images(input_dir, output_dir, num_variations=2):
    """
    Create enhanced composite artistic images for all images in directory
    
    Args:
        input_dir (str): Input directory with images
        output_dir (str): Output directory
        num_variations (int): Number of style variations per image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    enhanced_styles = ["high_coverage_splashes", "complex_geometric", "dense_paint_strokes", 
                      "vibrant_organic", "layered_artistic"]
    
    print(f"Creating enhanced composite artistic images for {len(image_files)} images")
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        
        for i in range(num_variations):
            style = random.choice(enhanced_styles)
            opacity = random.uniform(0.75, 0.95)
            preserve_face = random.choice([True, False])
            
            output_filename = f"{image_name}_enhanced_{style}_{i+1}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            create_enhanced_composite_artistic_image(image_file, output_path, style, opacity, preserve_face)

def generate_demo_enhanced_masks(output_dir="./demo_enhanced"):
    """Generate demonstration enhanced composite artistic masks"""
    os.makedirs(output_dir, exist_ok=True)
    
    enhanced_styles = ["high_coverage_splashes", "complex_geometric", "dense_paint_strokes", 
                      "vibrant_organic", "layered_artistic"]
    
    for style in enhanced_styles:
        mask, color_overlay = generate_enhanced_artistic_mask(256, 256, style)
        
        # Save mask
        mask_filename = f"demo_enhanced_{style}_mask.png"
        cv2.imwrite(os.path.join(output_dir, mask_filename), mask)
        
        # Save color overlay
        overlay_filename = f"demo_enhanced_{style}_overlay.png"
        cv2.imwrite(os.path.join(output_dir, overlay_filename), color_overlay)
    
    print(f"Demo enhanced masks saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate inpainting masks')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['basic', 'composite', 'enhanced', 'demo'],
                       help='Mask generation mode')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for masks')
    parser.add_argument('--mask_type', type=str, default='irregular',
                       choices=['irregular', 'rectangular', 'circular', 'stroke', 'mixed'],
                       help='Type of mask to generate')
    parser.add_argument('--style', type=str, default='high_coverage_splashes',
                       choices=['colorful_splashes', 'geometric_overlay', 'paint_strokes', 'mixed_artistic',
                               'high_coverage_splashes', 'complex_geometric', 'dense_paint_strokes', 
                               'vibrant_organic', 'layered_artistic'],
                       help='Artistic style for composite masks')
    parser.add_argument('--num_variations', type=int, default=1,
                       help='Number of mask variations per image')
    parser.add_argument('--coverage', type=float, default=None,
                       help='Coverage percentage (0.1 to 0.4)')
    parser.add_argument('--opacity', type=float, default=0.85,
                       help='Opacity for composite masks (0.7 to 1.0)')
    parser.add_argument('--preserve_face', action='store_true',
                       help='Preserve facial features in composite masks')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'demo':
        if 'enhanced' in args.style or args.style in ['high_coverage_splashes', 'complex_geometric', 
                                                      'dense_paint_strokes', 'vibrant_organic', 'layered_artistic']:
            generate_demo_enhanced_masks(args.output_dir)
        else:
            generate_demo_artistic_masks(args.output_dir)
    
    elif args.mode == 'basic':
        generate_basic_masks(args.input_dir, args.output_dir, args.mask_type, args.coverage)
    
    elif args.mode == 'composite':
        if args.style in ['high_coverage_splashes', 'complex_geometric', 'dense_paint_strokes', 
                         'vibrant_organic', 'layered_artistic']:
            batch_create_enhanced_composite_images(args.input_dir, args.output_dir, args.num_variations)
        else:
            batch_create_composite_images(args.input_dir, args.output_dir, args.num_variations)
    
    elif args.mode == 'enhanced':
        batch_create_enhanced_composite_images(args.input_dir, args.output_dir, args.num_variations)
    
    print(f"Mask generation completed! Check {args.output_dir} for results.")

def generate_basic_masks(input_dir, output_dir, mask_type='irregular', coverage=None):
    """Generate basic inpainting masks for images in directory with same naming"""
    # Get image files
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    print(f"Generating basic {mask_type} masks for {len(image_files)} images")
    
    for image_file in tqdm(image_files, desc="Processing images"):
        # Load image to get dimensions
        image = cv2.imread(image_file)
        if image is None:
            continue
            
        height, width = image.shape[:2]
        # Get original filename with extension
        original_filename = os.path.basename(image_file)
        
        # Generate mask
        mask = generate_inpainting_mask(height, width, mask_type, coverage)
        
        # Save mask with same name as input
        mask_path = os.path.join(output_dir, original_filename)
        cv2.imwrite(mask_path, mask)

def batch_create_composite_images(input_dir, output_dir, num_variations=1):
    """Create composite artistic images for all images in directory with same naming"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    styles = ["colorful_splashes", "geometric_overlay", "paint_strokes", "mixed_artistic"]
    
    print(f"Creating composite artistic images for {len(image_files)} images")
    
    for image_file in tqdm(image_files, desc="Processing images"):
        # Get original filename with extension
        original_filename = os.path.basename(image_file)
        image_name = os.path.splitext(original_filename)[0]
        image_ext = os.path.splitext(original_filename)[1]
        
        for i in range(num_variations):
            style = random.choice(styles)
            
            # Use same name as input image (with extension)
            if num_variations == 1:
                output_filename = original_filename
            else:
                # Only add number if multiple variations
                output_filename = f"{image_name}_{i+1}{image_ext}"
            
            output_path = os.path.join(output_dir, output_filename)
            
            create_composite_artistic_image(image_file, output_path, style)
def create_composite_artistic_image(image_path, output_path, style="colorful_splashes"):
    """Create composite image with artistic masks with complete opacity (basic version)"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return False
        
        # Resize if needed
        original_size = image.shape[:2]
        if original_size != (256, 256):
            image = cv2.resize(image, (256, 256))
        
        # Generate artistic composite mask
        mask, color_overlay = generate_artistic_composite_mask(256, 256, style)
        
        # Apply composite mask with complete opacity
        result = apply_composite_mask(image, mask, color_overlay, opacity=1.0)
        
        # Resize back to original if needed
        if original_size != (256, 256):
            result = cv2.resize(result, (original_size[1], original_size[0]))
        
        # Save result
        cv2.imwrite(output_path, result)
        return True
        
    except Exception as e:
        print(f"Error creating composite image: {e}")
        return False
def apply_composite_mask(image, mask, color_overlay, opacity=1.0):
    """Apply composite mask with complete opacity (basic version)"""
    result = image.copy().astype(np.float32)
    overlay = color_overlay.astype(np.float32)
    
    # Complete opacity blending
    mask_norm = (mask > 0).astype(np.float32)
    mask_3d = np.stack([mask_norm] * 3, axis=2)
    
    final_result = image.astype(np.float32) * (1 - mask_3d * opacity) + overlay * mask_3d * opacity
    
    return np.clip(final_result, 0, 255).astype(np.uint8)

def batch_create_enhanced_composite_images(input_dir, output_dir, num_variations=2):
    """
    Create enhanced composite artistic images for all images in directory with same naming
    
    Args:
        input_dir (str): Input directory with images
        output_dir (str): Output directory
        num_variations (int): Number of style variations per image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    enhanced_styles = ["high_coverage_splashes", "complex_geometric", "dense_paint_strokes", 
                    "vibrant_organic", "layered_artistic"]
    
    print(f"Creating enhanced composite artistic images for {len(image_files)} images")
    
    for image_file in tqdm(image_files, desc="Processing images"):
        # Get original filename with extension
        original_filename = os.path.basename(image_file)
        image_name = os.path.splitext(original_filename)[0]
        image_ext = os.path.splitext(original_filename)[1]
        
        for i in range(num_variations):
            style = random.choice(enhanced_styles)
            opacity = 1.0  # Set to complete opacity
            preserve_face = random.choice([True, False])
            
            # Use same name as input image (with extension)
            if num_variations == 1:
                output_filename = original_filename
            else:
                # Only add number if multiple variations
                output_filename = f"{image_name}_{i+1}{image_ext}"
            
            output_path = os.path.join(output_dir, output_filename)
            
            create_enhanced_composite_artistic_image(image_file, output_path, style, opacity, preserve_face)

def create_enhanced_composite_artistic_image(image_path, output_path, style="high_coverage_splashes", 
                                           opacity=1.0, preserve_face=True):
    """
    Create enhanced composite image with artistic masks - now with complete opacity
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save output
        style (str): Enhanced artistic style
        opacity (float): Blending opacity set to 1.0 for complete opacity
        preserve_face (bool): Whether to preserve facial features
        
    Returns:
        bool: Success status
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return False
        
        # Resize if needed
        original_size = image.shape[:2]
        if original_size != (256, 256):
            image = cv2.resize(image, (256, 256))
        
        # Generate enhanced artistic composite mask
        mask, color_overlay = generate_enhanced_artistic_mask(256, 256, style)
        
        # Apply enhanced composite mask with complete opacity
        result = apply_enhanced_composite_mask(image, mask, color_overlay, opacity, preserve_face)
        
        # Resize back to original if needed
        if original_size != (256, 256):
            result = cv2.resize(result, (original_size[1], original_size[0]))
        
        # Save result
        cv2.imwrite(output_path, result)
        print(f"Enhanced composite artistic image saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating enhanced composite image: {e}")
        return False
    
def apply_enhanced_composite_mask(image, mask, color_overlay, opacity=1.0, preserve_face=True):
    """
    Apply enhanced composite mask with complete opacity and face preservation
    
    Args:
        image (numpy.ndarray): Original image (BGR)
        mask (numpy.ndarray): Binary mask
        color_overlay (numpy.ndarray): Color overlay (BGR)
        opacity (float): Opacity set to 1.0 for complete opaque mask
        preserve_face (bool): Whether to preserve facial features
        
    Returns:
        numpy.ndarray: Image with enhanced composite mask applied
    """
    result = image.copy().astype(np.float32)
    overlay = color_overlay.astype(np.float32)
    
    # Create face preservation mask if requested
    if preserve_face:
        height, width = image.shape[:2]
        face_center_x, face_center_y = width // 2, height // 2
        face_radius = min(width, height) // 4
        
        preservation_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Preserve eye regions
        eye_y = int(face_center_y - face_radius * 0.2)
        cv2.circle(preservation_mask, (int(face_center_x - face_radius * 0.3), eye_y), 
                   face_radius // 5, 255, -1)
        cv2.circle(preservation_mask, (int(face_center_x + face_radius * 0.3), eye_y), 
                   face_radius // 5, 255, -1)
        
        # Preserve mouth region
        mouth_y = int(face_center_y + face_radius * 0.3)
        cv2.ellipse(preservation_mask, (face_center_x, mouth_y), 
                    (face_radius // 3, face_radius // 6), 0, 0, 360, 255, -1)
        
        # Reduce mask intensity in preservation areas
        mask_float = mask.astype(np.float32) / 255.0
        preservation_float = preservation_mask.astype(np.float32) / 255.0
        mask_float = mask_float * (1 - preservation_float * 0.7)  # Reduce by 70% in face areas
        mask = (mask_float * 255).astype(np.uint8)
    
    # Enhanced overlay blend with multiple blend modes - now fully opaque
    mask_norm = (mask > 0).astype(np.float32)
    
    # Color dodge effect for vibrant colors
    for c in range(3):
        base = result[:, :, c] / 255.0
        blend = overlay[:, :, c] / 255.0
        
        # Enhanced color dodge with saturation boost
        dodge_result = np.where(
            blend < 1.0,
            np.minimum(1.0, base / (1.001 - blend)),  # Avoid division by zero
            1.0
        )
        
        # Screen blend for additional brightness
        screen_result = 1 - ((1 - base) * (1 - blend))
        
        # Combine dodge and screen
        combined = (dodge_result * 0.6 + screen_result * 0.4)
        result[:, :, c] = combined * 255
    
    # Apply complete opacity blending (opacity = 1.0)
    mask_3d = np.stack([mask_norm] * 3, axis=2)
    final_result = image.astype(np.float32) * (1 - mask_3d * opacity) + result * mask_3d * opacity
    
    # Convert to HSV for saturation enhancement
    final_result_hsv = cv2.cvtColor(final_result.astype(np.uint8), cv2.COLOR_BGR2HSV)
    final_result_hsv[:, :, 1] = np.clip(final_result_hsv[:, :, 1] * 1.2, 0, 255)  # Boost saturation
    final_result = cv2.cvtColor(final_result_hsv, cv2.COLOR_HSV2BGR)
    
    return np.clip(final_result, 0, 255).astype(np.uint8)

def generate_demo_artistic_masks(output_dir="./demo_artistic"):
    """Generate demonstration artistic masks"""
    os.makedirs(output_dir, exist_ok=True)
    
    styles = ["colorful_splashes", "geometric_overlay", "paint_strokes", "mixed_artistic"]
    
    for style in styles:
        mask, color_overlay = generate_artistic_composite_mask(256, 256, style)
        
        # Save mask
        mask_filename = f"demo_{style}_mask.png"
        cv2.imwrite(os.path.join(output_dir, mask_filename), mask)
        
        # Save color overlay
        overlay_filename = f"demo_{style}_overlay.png"
        cv2.imwrite(os.path.join(output_dir, overlay_filename), color_overlay)
    
    print(f"Demo artistic masks saved to {output_dir}")

if __name__ == "__main__":
    main()