import os
import os.path as ops
import imageio
import cv2


def folder_to_vid(folder_path, output_file_path):
    writer1 = imageio.get_writer(output_file_path, fps=60)
    files = next(os.walk(folder_path))[0]
    for file in files:
        im = cv2.imread(f"{folder_path}/{file}")
        im = cv2.resize(im, (1920, 1080))
        writer1.append_data(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    writer1.close()

if __name__ == "__main__":
    folder_to_vid("./images", "./video.mp4")