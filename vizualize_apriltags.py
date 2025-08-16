import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

def draw_circles_on_images(csv_file, image_dirs):
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Get unique timestamps
    timestamps = df['timestamp_ns'].unique()
    
    for timestamp in timestamps:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for cam_id in range(3):  # Assuming cam_id values are 0, 1, 2
            image_path = os.path.join(image_dirs[cam_id], f"{timestamp}.jpg")  # Adjust extension if needed
            
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found!")
                continue
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading image {image_path}")
                continue
            
            # Convert to RGB for displaying with matplotlib
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Filter rows for current cam_id and timestamp
            cam_df = df[(df['timestamp_ns'] == timestamp) & (df['cam_id'] == cam_id)]
            
            for _, row in cam_df.iterrows():
                x, y, radius, corner_id = int(row['corner_x']), int(row['corner_y']), int(row['radius']), int(row['corner_id'])
                cv2.circle(img, (x, y), radius, (255, 0, 0), 2)  # Draw blue circles
                cv2.putText(img, str(corner_id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            axes[cam_id].imshow(img)
            axes[cam_id].set_title(f"Camera {cam_id}")
            axes[cam_id].axis('off')
        
        plt.show()

# Example usage
image_dirs = {
    0: "/home/jake/calibration_euroc_data_copy/mav0/cam0/data/",
    1: "/home/jake/calibration_euroc_data_copy/mav0/cam1/data/",
    2: "/home/jake/calibration_euroc_data_copy/mav0/cam2/data/"
}
draw_circles_on_images("/home/jake/atest/good_corners.csv", image_dirs)