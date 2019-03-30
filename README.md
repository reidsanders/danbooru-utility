# Danbooru Utility 

Danbooru Utility is a simple python script for working with gwern's Danbooru2018 dataset. It can explore the dataset, filter by tags, rating, and score, detect faces, and resize the images. I've been using it to make datasets for gan training.


## Install

```sh
pip3 install danbooru-utility
```

Make sure you have downloaded [Danbooru2018](https://www.gwern.net/Danbooru2018). It's ~3.3M annotated anime images, so downloading may take a long time.

## Usage

First let's search for something particular.

```sh
$ danbooru-utility \
--directory ~/datasets/danbooru-gwern/danbooru2018/ \
 --rating "s" \
--required_tags "archer,toosaka_rin,hug" \
--banned_tags "photo,comic" \
--max_examples 3 \
--img_size 256 \
--faces False

Processed 3 files. Added 3 images. It took 14.39 sec
```

This will find three images with the required tags, and without the banned tags and resize them to 256x256. Note this took a long time since the filtering is just done in a loop. Let's check what this produced in `out-images`:

![Rin Archer Example](./img/rin_archer1.jpg)
![Rin Archer Example](./img/rin_archer2.jpg)
![Rin Archer Example](./img/rin_archer3.png)

Now let's run the same command but with face detection:

```sh
$ danbooru-utility \
--directory ~/datasets/danbooru-gwern/danbooru2018/ \
 --rating "s" \
--required_tags "archer,toosaka_rin,hug" \
--banned_tags "photo,comic" \
--max_examples 3 \
--img_size 256 \
--faces True

Processed 3 files. Added 1 images. It took 12.48 sec
```

That produced:

![Rin Archer Example](./img/rin_archer_face_default.jpg)

So it zoomed in slightly in on the face.

Let's change the `face_scale` parameter. This controls how much of the image around the face is included in the crop.

```sh
$ danbooru-utility \
--directory ~/datasets/danbooru-gwern/danbooru2018/ \
 --rating "s" \
--required_tags "archer,toosaka_rin,hug" \
--banned_tags "photo,comic" \
--max_examples 3 \
--img_size 256 \
--faces True
--face_scale 1.8

Processed 3 files. Added 1 images. It took 12.49 sec
```

![Rin Archer Example](./img/rin_archer_face_scale.jpg)

A little tighter crop.

If you have already processed some images this utility will check and not reproduce them, so if you change the image processing parameters you will need to delete the already processed images or use a new `save_dir`. You can also specify a `link_dir` to symlink to. So you can, for instance, resize a large number of images, and then create datasets for specific tags quickly.

## Config

For details on parameters check help.

```sh
$ danbooru-utility -h
usage: danbooru-utility [-h] [-d DIRECTORY] [--metadata_dir METADATA_DIR]
                        [--save_dir SAVE_DIR] [--link_dir LINK_DIR]
                        [-r REQUIRED_TAGS] [-b BANNED_TAGS] [-a ATLEAST_TAGS]
                        [--ratings RATINGS] [--score_range SCORE_RANGE]
                        [-n ATLEAST_NUM] [--preview [PREVIEW]]
                        [--faces [FACES]] [--face_scale FACE_SCALE]
                        [--max_examples MAX_EXAMPLES] [--img_size IMG_SIZE]

danbooru2018 utility script

optional arguments:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        Danbooru directory Directory
  --metadata_dir METADATA_DIR
                        Metadata path below base directory. Will load all json
                        files here
  --save_dir SAVE_DIR   Directory processed images are saved to.
  --link_dir LINK_DIR   Directory with already processed images. Used to
                        symlink to if the files exist.
  -r REQUIRED_TAGS, --required_tags REQUIRED_TAGS
                        Tags required.
  -b BANNED_TAGS, --banned_tags BANNED_TAGS
                        Tags disallowed.
  -a ATLEAST_TAGS, --atleast_tags ATLEAST_TAGS
                        Requires some number of these tags.
  --ratings RATINGS     Only include images with these ratings. "s,q,e"
                        represent "safe,questionable,explicit"
  --score_range SCORE_RANGE
                        Only include images inside this score range
  -n ATLEAST_NUM, --atleast_num ATLEAST_NUM
                        Minimum number of atleast_tags required.
  --preview [PREVIEW]   preview images
  --faces [FACES]       detect faces and try to include them in top of image
  --face_scale FACE_SCALE
                        height and width multiplier over size of face
  --max_examples MAX_EXAMPLES
                        Maximum number of files to load.
  --img_size IMG_SIZE   Size of side for resized images.

```

Here's an example metadata entry in Danbooru2018:

```python
{'approver_id': '0',
 'created_at': '2016-10-26 09:32:42.38506 UTC',
 'down_score': '0',
 'favs': ['12082', '334419', '496852', '516035', '487870'],
 'file_ext': 'jpg',
 'file_size': '753165',
 'has_children': False,
 'id': '2524919',
 'image_height': '874',
 'image_width': '1181',
 'is_banned': False,
 'is_deleted': False,
 'is_flagged': False,
 'is_note_locked': False,
 'is_pending': False,
 'is_rating_locked': False,
 'is_status_locked': False,
 'last_commented_at': '1970-01-01 00:00:00 UTC',
 'last_noted_at': '1970-01-01 00:00:00 UTC',
 'md5': 'a9260780fbf5cfd661878f92a268124e',
 'parent_id': '2524918',
 'pixiv_id': '54348754',
 'pools': [],
 'rating': 's',
 'score': '3',
 'source': 'http://i3.pixiv.net/img-original/img/2015/12/31/13/31/23/54348754_p13.jpg',
 'tags': [{'category': '0', 'id': '540830', 'name': '1boy'},
		  {'category': '0', 'id': '470575', 'name': '1girl'},
		  {'category': '1', 'id': '1332557', 'name': 'akira_(ubw)'},
		  {'category': '4', 'id': '396', 'name': 'archer'},
		  {'category': '0', 'id': '13200', 'name': 'black_hair'},
		  {'category': '0', 'id': '3389', 'name': 'blush'},
		  {'category': '0', 'id': '4563', 'name': 'bow'},
		  {'category': '0', 'id': '465619', 'name': 'closed_eyes'},
		  {'category': '0', 'id': '71730', 'name': 'dark_skin'},
		  {'category': '0', 'id': '610236', 'name': 'dark_skinned_male'},
		  {'category': '3', 'id': '5', 'name': 'fate/stay_night'},
		  {'category': '3', 'id': '662939', 'name': 'fate_(series)'},
		  {'category': '0', 'id': '374938', 'name': 'frown'},
		  {'category': '0', 'id': '374844', 'name': 'hair_bow'},
		  {'category': '0', 'id': '5126', 'name': 'hug'},
		  {'category': '0', 'id': '1815', 'name': 'smile'},
		  {'category': '0', 'id': '125238', 'name': 'sweatdrop'},
		  {'category': '4', 'id': '400140', 'name': 'toosaka_rin'},
		  {'category': '0', 'id': '652604', 'name': 'two_side_up'},
		  {'category': '0', 'id': '16581', 'name': 'white_hair'}],
 'up_score': '3',
 'updated_at': '2018-06-05 05:37:49.87865 UTC',
 'uploader_id': '39276'}

```

## Improvements

This currently only filter using tags, rating, and score, but there's more info to use.

This could load the dataset into a relational database, allowing much more efficient and powerful querying.

The face detection has room for improvement. It has rare false positives, and a fair number of false negatives.

## Acknowledgements

Thanks to [gwern](https://gwern.net) for the excellent danbooru dataset.

Thanks to [nagadomi](https://github.com/nagadomi/lbpcascade_animeface) for the anime face detection model.

## License

MIT License

Copyright (c) 2019 Reid Sanders

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
