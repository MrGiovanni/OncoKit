'''
python -W ignore create_dataset_videos.py --datapath data_demo --videopath /Users/zongwei.zhou/Desktop/AbdomenAtlasVideo --resizedX 120 --resizedY 120 --num_core 2
'''

import os
import argparse
from tqdm import tqdm 
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import nibabel as nib 
import numpy as np
from PIL import Image
import cv2
import imageio
def full_make_png(case_name, args):
    
    for plane in ['axial', 'coronal', 'sagittal']:
        if not os.path.exists(os.path.join(args.png_save_path, plane, case_name)):
            os.makedirs(os.path.join(args.png_save_path, plane, case_name))

    image_name = f'ct.nii.gz'

    image_path = os.path.join(args.datapath, case_name, image_name)

    # single case
    image = nib.load(image_path).get_fdata().astype(np.int16)
    
    # change orientation
    ornt = nib.orientations.axcodes2ornt(('F', 'L', 'U'), (('L','R'),('B','F'),('D','U')))
    image = nib.orientations.apply_orientation(image, ornt)
    
    image[image > args.high_range] = args.high_range
    image[image < args.low_range] = args.low_range
    image = np.round((image - args.low_range) / (args.high_range - args.low_range) * 255.0).astype(np.uint8)
    image = np.repeat(image.reshape(image.shape[0],image.shape[1],image.shape[2],1), 3, axis=3)
    
    for z in range(image.shape[2]):
        if args.resizedX > 0 or args.resizedY > 0:
            resized_slices = cv2.resize(image[:,:,z,:], 
                                        (args.resizedX, args.resizedY), 
                                        interpolation=cv2.INTER_AREA,
                                        )
            Image.fromarray(resized_slices).save(os.path.join(args.png_save_path, 'axial', case_name, str(z)+'.png'))
        else:
            Image.fromarray(image[:,:,z,:]).save(os.path.join(args.png_save_path, 'axial', case_name, str(z)+'.png'))

    for z in range(image.shape[1]):
        Image.fromarray(image[:,z,:,:]).save(os.path.join(args.png_save_path, 'sagittal', case_name, str(z)+'.png'))

    for z in range(image.shape[0]):
        Image.fromarray(image[z,:,:,:]).save(os.path.join(args.png_save_path, 'coronal', case_name, str(z)+'.png'))

def make_avi(case_name, plane, args):

    if not os.path.exists(os.path.join(args.avi_save_path, plane)):
        os.makedirs(os.path.join(args.avi_save_path, plane))
    if not os.path.exists(os.path.join(args.gif_save_path, plane)):
        os.makedirs(os.path.join(args.gif_save_path, plane))

    image_folder = os.path.join(args.png_save_path, plane, case_name)
    video_name = os.path.join(args.avi_save_path, plane, case_name+'.avi')
    gif_name = os.path.join(args.gif_save_path, plane, case_name+'.gif')
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    for i in range(len(images)):
        images[i] = images[i].replace('.png','')
        images[i] = int(images[i])
    images.sort()

    frame = cv2.imread(os.path.join(image_folder, str(images[0])+'.png'))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, args.FPS, (width,height))

    imgs = []
    for image in images:
        img = cv2.imread(os.path.join(image_folder, str(image)+'.png'))
        video.write(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    cv2.destroyAllWindows()
    video.release()
    imageio.mimsave(gif_name, imgs, duration=args.FPS*0.4)

def event(pid, args):

    full_make_png(pid, args)
    # for plane in ['axial', 'coronal', 'sagittal']:
    for plane in ['axial']:
        make_avi(pid, plane, args)

def main(args):

    if not os.path.exists(args.videopath):
        os.makedirs(args.videopath)

    args.png_save_path = os.path.join(args.videopath, 'pngs')
    if not os.path.exists(args.png_save_path):
        os.makedirs(args.png_save_path)
    
    args.gif_save_path = os.path.join(args.videopath, 'gifs')
    if not os.path.exists(args.gif_save_path):
        os.makedirs(args.gif_save_path)

    args.avi_save_path = os.path.join(args.videopath, 'avis')
    if not os.path.exists(args.avi_save_path):
        os.makedirs(args.avi_save_path)

    folder_names = [name for name in os.listdir(args.datapath) if os.path.isdir(os.path.join(args.datapath, name))]
    print('>> Converting {} cases.'.format(len(folder_names)))

    if args.num_core > 0:
        num_core = args.num_core
    else:
        num_core = int(cpu_count())

    print('>> {} CPU cores are secured.'.format(num_core))

    with ProcessPoolExecutor(max_workers=num_core) as executor:

        futures = {executor.submit(event, pid, args): pid for pid in folder_names}

        for future in tqdm(as_completed(futures), total=len(futures), ncols=80):
            folder = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {folder}: {e}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, default='/mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/', help='Path to the dataset.')
    parser.add_argument('--videopath', type=str, default='/mnt/T8/error_analysis/', help='Path to save the videos.')
    parser.add_argument('--num_core', type=int, default=0, help='Number of CPU cores to use.')
    parser.add_argument('--low_range', type=int, default=-150, help='Low range of the CT image.')
    parser.add_argument('--high_range', type=int, default=200, help='High range of the CT image.')
    parser.add_argument('--FPS', type=int, default=20, help='Frames per second of the video.')
    parser.add_argument('--resizedX', type=int, default=0, help='Resized X dimension.')
    parser.add_argument('--resizedY', type=int, default=0, help='Resized Y dimension.')

    args = parser.parse_args()

    main(args)