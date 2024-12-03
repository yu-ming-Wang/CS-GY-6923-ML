import cv2
import os
import zipfile
import shutil
import logging
from use_video_to_image_zip import frame_extract, crop_to_square, long_time_task

cv2.setNumThreads(1)

logging.basicConfig(filename='conversion.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')

def get_video_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.mp4')]

def split_list(input_list, n_parts):
    avg = len(input_list) / float(n_parts)
    out = []
    last = 0.0

    while last < len(input_list):
        out.append(input_list[int(last):int(last + avg)])
        last += avg

    return out

def zip_images(folder_name, output_dir):
    try:
        zip_filename = os.path.join(output_dir, f"{os.path.basename(folder_name)}.zip")
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_name):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_name))
        shutil.rmtree(folder_name)
    except Exception as e:
        logging.error(f"Error zipping folder {folder_name}: {e}")
        raise

def process_videos(video_list, input_dir, output_dir):
    for video in video_list:
        try:
            video_path = os.path.join(input_dir, video)
            output_folder = frame_extract(video_path, output_dir)
            zip_images(output_folder, output_dir)
            logging.info(f"Processed and zipped: {video}")
        except Exception as e:
            logging.error(f"Error processing video {video}: {e}")

def main():
    input_dir = '/scratch/jl10897/DeepPersonality-main/datasets/MOSE/new_train'
    output_dir = '/scratch/jl10897/DeepPersonality-main/datasets/MOSE/new_train_data'
    
    try:
        videos = get_video_files(input_dir)
        video_batches = split_list(videos, 10)  # Split into 10 batches to control thread usage

        for batch in video_batches:
            process_videos(batch, input_dir, output_dir)
    except Exception as e:
        logging.error(f"Error in main processing loop: {e}")

if __name__ == "__main__":
    main()
