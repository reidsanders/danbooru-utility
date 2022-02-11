# !/usr/bin/python
# -*- coding: utf-8 -*-
#
import sys
import os
import time
import argparse
from pprint import pprint as pp
import json
from PIL import Image
import pathlib
from resizeimage import resizeimage
import zipfile
import cv2
import numpy as np
import tempfile
import multiprocessing as mp
from itertools import chain

NCORE = mp.cpu_count()


def str2bool(v):
    """Converts strings to appropriate bool, for argparse."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print(f'ARG: {v}')
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args(arg_input):
    """Takes args input and returns them as a argparse parser

    Parameters 
    -------------

    arg_input : list, shape (n_nargs,)
        contains list of arguments passed to function

    Returns
    -------------

    args : namespace
        contains namespace with keys and values for each parser argument

    """
    parser = argparse.ArgumentParser(description='danbooru2018 utility script')
    parser.add_argument(
        '-d',
        '--directory',
        type=str,
        default='danbooru2018',
        help='Danbooru dataset directory.',
    )
    parser.add_argument(
        '--metadata_dir',
        type=str,
        default='metadata',
        help='Metadata path below base directory. Will load all json files here.',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='out-images',
        help='Directory processed images are saved to.',
    )
    parser.add_argument(
        '--link_dir',
        type=str,
        default='link-images',
        help=
        'Directory with already processed images. Used to symlink to if the files exist.',
    )
    parser.add_argument(
        '-r',
        '--required_tags',
        type=lambda s: {str(item) for item in s.split(',')},
        default=set(),
        help='Tags required.',
    )
    parser.add_argument(
        '-b',
        '--banned_tags',
        type=lambda s: {str(item) for item in s.split(',')},
        default='',
        help='Tags disallowed.',
    )
    parser.add_argument(
        '-a',
        '--atleast_tags',
        type=lambda s: {str(item) for item in s.split(',')},
        default='',
        help='Requires some number of these tags.',
    )
    parser.add_argument(
        '--ratings',
        type=lambda s: {str(item) for item in s.split(',')},
        default='s,q,e',
        help='Only include images with these ratings.\
        "s,q,e" are the possible entries, and represent "safe,questionable,explicit".',
    )
    parser.add_argument(
        '--score_range',
        type=lambda s: [int(item) for item in s.split(',')],
        default='-999999999,999999999',
        help='Only include images inside this score range.',
    )
    parser.add_argument(
        '-n',
        '--atleast_num',
        type=int,
        default=0,
        help='Minimum number of atleast_tags required.',
    )
    parser.add_argument(
        '--overwrite',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Overwrite images in save directory.'
    )
    parser.add_argument(
        '--preview',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Preview images.'
    )
    parser.add_argument(
        '--faces',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Detect faces and try to include them in top of image.'
    )
    parser.add_argument(
        '--face_scale',
        type=float,
        default=2.5,
        help='Height and width multiplier over size of face.',
    )
    parser.add_argument(
        '--max_examples',
        type=int,
        default=1e15,
        help='Maximum number of files to load.',
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='Size of side for resized images.',
    )

    args = parser.parse_args(arg_input)
    return args


def tag_check(
    tags,
    rating,
    score,
    score_range=[-1e15,1e15],
    included_ratings=set(),
    banned_tags=set(),
    required_tags=set(),
    atleast_tags=set(),
    atleast_num=0,
):
    """
    checks if tags and rating match requirements
    """
    if banned_tags & tags:
        return False

    if required_tags & tags != required_tags:
        return False

    if len(atleast_tags & tags) < atleast_num:
        return False

    if rating not in included_ratings:
        return False

    if int(score) < score_range[0]:
        return False

    if int(score) > score_range[1]:
        return False

    return True


def load_data(args):
    """
    loads and yields image data from metadata files
    """
    data = []
    metadata_paths = find_metadata_files(
        os.path.join(args.directory, args.metadata_dir)
    )
    i = 0
    for path in metadata_paths:
        with open(path, "r") as f:
            for line in f:
                #if i >= args.max_examples:
                    #break
                example = json.loads(line)
                if tag_check(
                    {y['name']
                     for y in example['tags']},
                    example['rating'],
                    example['score'],
                    score_range=args.score_range,
                    included_ratings=args.ratings,
                    required_tags=args.required_tags,
                    banned_tags=args.banned_tags,
                    atleast_tags=args.atleast_tags,
                    atleast_num=args.atleast_num,
                ):
                    i += 1
                    yield example


def find_metadata_files(directory):
    """
    finds metadata files from directory
    """
    p = pathlib.Path(directory).glob('**/*')
    files = [x for x in p if x.is_file()]
    return files


def detect_faces_tasker(task_queue, num_processed_return, metadata_return, iolock):
    """
    loads image data from task_queue and runs face detection
    """
    num_processed = 0
    metadata = []
    while True:
        args = task_queue.get()
        if args is None:
            break
        num_processed_inside, metadata_inside = detect_faces(*args)
        num_processed += num_processed_inside
        metadata += metadata_inside
    num_processed_return.append(num_processed)
    metadata_return += metadata


def resize_and_save_images_mp(data_gen, args):
    """
    Loops over loaded image data and resizes and saves if the tags match requirements. 
    Detects faces if that was selected, using multiprocessing for face detection. 

    """
    start_time = time.time()
    i = 0 # number of files
    num_processed = 0 # number of faces processed
    manager = mp.Manager() # Manager to allow shared variables
    task_queue = manager.Queue(maxsize=NCORE) # queue of images for face detection processes
    num_processed_return = manager.list()
    metadata_return = manager.list()
    iolock = mp.Lock()
    pool = mp.Pool(
        NCORE,
        initializer=detect_faces_tasker,
        initargs=(task_queue, num_processed_return, metadata_return, iolock)
    )
    metadata = []
    for example in data_gen:
        if i >= args.max_examples:
            break
        img_id = example['id']
        try:
            write_file = img_id + '.' + example['file_ext']
            # in danbooru201* the files are split into different directories based on mod of image id 
            load_path = os.path.join(
                args.directory,
                'original',
                "{0:0{width}}".format(int(img_id) % 1000, width=4),
                "{0}.{1}".format(img_id, example['file_ext']),
            )
            # Extract and process zipfiles
            if example['file_ext'] == 'zip':
                try:
                    zip_ref = zipfile.ZipFile(load_path, 'r')
                    tmp_dir = tempfile.mkdtemp()
                    zip_ref.extractall(path=tmp_dir)
                except:
                    print("Failed to open zip file")
                    continue
                for name in zip_ref.namelist():
                    load_path = os.path.join(tmp_dir, name)
                    write_file = img_id + '_' + name
                    if args.faces:
                        task_queue.put(
                            (
                                load_path, write_file, args.save_dir,
                                args.link_dir, args.img_size, args.face_scale,
                                args.overwrite, example,
                            )
                        )
                    else:
                        num_processed_inside = resize_and_save_image(
                            load_path, write_file, args.save_dir, args.link_dir,
                            args.img_size, args.overwrite
                        )
                        num_processed += num_processed_inside
                        if num_processed_inside > 0:
                            example["filename"] = write_file
                            metadata.append(example)
                    i += 1
                    if i % 100 == 0:
                        print(
                                f'Processed {i} files. It took {time.time() - start_time:.2f} sec'
                        )
                    if i >= args.max_examples:
                        break
                zip_ref.close()
                continue
            # If not a zipfile, process image
            if args.faces:
                task_queue.put(
                    (
                        load_path, write_file, args.save_dir, args.link_dir,
                        args.img_size, args.face_scale,
                        args.overwrite, example,
                    )
                )
            else:
                num_processed_inside = resize_and_save_image(
                    load_path, write_file, args.save_dir, args.link_dir,
                    args.img_size, args.overwrite
                )
                num_processed += num_processed_inside
                if num_processed_inside > 0:
                    example["filename"] = write_file
                    metadata.append(example)
            i += 1
            if i % 100 == 0:
                print(
                        f'Processed {i} files. It took {time.time() - start_time:.2f} sec'
                )
        except (FileNotFoundError, OSError) as detail:
            print(f"Unable to load image {detail}")
    
    # Send end task sentinel
    for _ in range(NCORE):
        task_queue.put(None)
    pool.close()
    pool.join()
    num_processed += sum(num_processed_return)
    metadata += list(chain(metadata_return))
    jsonfile = os.path.join(args.save_dir, "index.json")
    with open(jsonfile, "w+") as f:
        print(f"Saving JSON metadata file: {jsonfile}")
        jsondata = json.dumps(
            {
                "data": metadata,
                # Add other global metadata here
            }
        )
        f.write(jsondata)
    
    print(
            f'\nProcessed {i} files. Added {num_processed} images. It took {time.time() - start_time:.2f} sec'
    )


def resize_and_save_image(load_path, write_file, save_dir, link_dir, img_size, overwrite):
    """
    Resize image and save with white border if not square

    """
    write_path = os.path.join(save_dir, write_file)
    link_path = os.path.join(link_dir, write_file)
    if not overwrite:
        if exists_or_link(write_path, link_path):
            return 1
    try:
        img = Image.open(load_path)
    except Exception as detail:
        print("Failed to open image: {}".format(detail))
        return 0
    try:
        if img_size >= 0:
            img = resizeimage.resize_contain(img, [img_size, img_size])
        if img.mode in ("RGBA", "P"): 
            img = img.convert("RGB")
        img.save(write_path, img.format)
        img.close()

    except Exception as detail:
        print("Failed in resize and save img: {}".format(detail))
        return 0
    return 1


def exists_or_link(write_path, link_path):
    """
    checks if file already exists,
    or is in link directory, adds symlink to it if so
    """
    if os.path.exists(write_path):
        return True
    if os.path.exists(link_path):
        os.symlink(link_path, write_path)
        return True
    return False


def detect_faces(
    load_path,
    write_file,
    save_dir,
    link_dir,
    img_size,
    face_scale,
    overwrite,
    info,
    cascade_file_name="lbpcascade_animeface.xml"
):
    """
    Detect faces in image, and crops to fit them in upper part of image

    """
    cascade_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), cascade_file_name)
    num_processed = 0
    metadata = []
    # Check if already processed. Checks for up to 20 faces
    for i in range(20):
        face_write_file = f"face{i}_{write_file}"
        face_write_path = os.path.join(save_dir, face_write_file)
        face_link_path = os.path.join(link_dir, face_write_file)
        if not overwrite:
            if exists_or_link(face_write_path, face_link_path):
                num_processed += 1
            elif num_processed > 0:
                return num_processed, metadata
    # If not sufficiently processed or linkable set up face recognition
    num_processed = 0
    if not os.path.isfile(cascade_file):
        raise RuntimeError(f"{cascade_file}: not found")

    cascade = cv2.CascadeClassifier(cascade_file)
    try:
        image = np.asarray(Image.open(load_path).convert('RGB'))
    except Exception as detail:
        print(f"Image Error: {detail}")
        return 0, []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)
    )
    for (x, y, w, h) in faces:
        cropw = int(w * face_scale)
        croph = int(h * face_scale)
        cropx = int(x - (cropw - w) / 2)
        cropy = int(y - (croph - h) / 4)

        if (cropx < 0):
            cropx = 0
        if (cropy < 0):
            cropy = 0

        crop_img = image[cropy:cropy + croph, cropx:cropx + cropw]
        if crop_img.shape[0] < img_size or crop_img.shape[1] < img_size:
            continue
        img = Image.fromarray(crop_img)
        face_write_file = f"face{num_processed}_{write_file}"
        face_write_path = os.path.join(save_dir, face_write_file)
        face_link_path = os.path.join(link_dir, face_write_file)

        try:
            img = resizeimage.resize_contain(img, [img_size, img_size])
            if img.mode in ("RGBA", "P"): 
                img = img.convert("RGB")
            img.save(face_write_path, img.format)
            num_processed += 1
            info["filename"] = face_write_file
            metadata.append(info)
        except Exception as detail:
            print(f"Image Error: {detail}")
    return num_processed, metadata


def preview(data_gen, args):
    """
    Display image and meta data

    """
    i = 0
    for example in data_gen:
        img_id = example['id']
        load_path = os.path.join(
            args.directory,
            'original',
            "{0:0{width}}".format(int(img_id) % 1000, width=4),
            "{0}.{1}".format(img_id, example['file_ext']),
        )
        try:
            img = Image.open(load_path)
            pp(example)
            img.show()
            input('Press enter for next img')
            img.close()
        except Exception as detail:
            print(f"Unable to load image {detail}")
        i += 1
        if i >= args.max_examples:
            break


def main(args=None):
    if args == None:
        arg_input = sys.argv[1:]
        args = get_args(arg_input)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    data_gen = load_data(args)
    resize_and_save_images_mp(data_gen, args)
    if args.preview:
        data_gen = load_data(args)
        preview(data_gen, args)


if __name__ == '__main__':
    main()
