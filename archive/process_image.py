import time
import os
import sys
from PIL import Image
import psutil

def process_image(input_path, output_path):
    print(f"--- Container Execution Started: Processing {os.path.basename(input_path)} ---")
    start_time = time.time()
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return
    
    img = Image.open(input_path)
    print(f"   Original Size: {img.size}")
    
    # Simulate heavy workload by doing multiple transformations
    # (resizing, cropping, and filtering)
    for i in range(1, 101):
        # Alternate resizing to make it compute-intensive
        size = (img.width // (i % 5 + 1), img.height // (i % 5 + 1))
        # Keep it at least 100x100
        size = (max(100, size[0]), max(100, size[1]))
        
        temp_img = img.resize(size, Image.LANCZOS)
        
        # Crop an area
        crop_box = (10, 10, min(img.width-10, 200), min(img.height-10, 200))
        crop_img = img.crop(crop_box)
        
        if i % 25 == 0:
            print(f"      Iteration {i}/100 complete...")

    # Final Save
    final_img = img.resize((512, 512), Image.LANCZOS)
    final_img.save(output_path)
    
    duration = time.time() - start_time
    print(f"--- Container Execution Finished: Success ({duration:.3f} seconds) ---")
    
    # Collect some basic metrics for reporting
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem_usage = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"METRICS_REPORT: DURATION={duration:.3f}, CPU={cpu_percent}, MEM={mem_usage:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_image.py <input> <output>")
    else:
        process_image(sys.argv[1], sys.argv[2])
