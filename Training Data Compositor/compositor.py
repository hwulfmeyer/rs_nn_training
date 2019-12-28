import PIL
import numpy.random as nr
from PIL import Image, ImageDraw, ImageFilter, ImageMath, ImageEnhance
import tensorflow as tf
from object_detection.utils import dataset_util

from Utils.file_utils import *

def is_crop_supported(meta,filters):
    for k in filters:
        if k not in meta: continue
        if meta[k] not in filters[k]:
            return False
    return True

def is_background_supported(dataset,filters):
    return dataset in filters["background_set"]

def balanced_choice(balanced_list):
    l = balanced_list
    while isinstance(l,list):
        if len(l)==1:
            l = l[0]
        else:
            l = nr.choice(l)
    return l

def generate_balanced_object_list(object_path,filters):
    typeObjectList = []
    typeObjectDict = {}
    metaFileList = get_recursive_file_list(object_path, file_matchers=['meta.json'])
    for metaFile in metaFileList:
        meta = load_json(metaFile)
        directory = metaFile.replace('/meta.json','')
        img_files = get_recursive_file_list(directory, file_extensions=['.png'])
        for f in img_files:
            if meta['identification'] == "undefined": continue
            if not is_crop_supported(meta,filters): continue
            meta['crop'] = f
            key1 = meta['robot_type']
            key2 = meta['identification']
            # important for balancing classes
            if key1 not in typeObjectDict: typeObjectDict[key1] = {}
            if key2 not in typeObjectDict[key1]: typeObjectDict[key1][key2] = []
            typeObjectDict[key1][key2].append(meta)
    for key1 in typeObjectDict:
        typeObjectList.append([])
        for key2 in typeObjectDict[key1]:
            typeObjectList[-1].append(typeObjectDict[key1][key2])
    return typeObjectList

def generate_balanced_background_list(background_path,filters):
    backgroundFileList = []
    for d in os.listdir(background_path):
        if not is_background_supported(d,filters): continue
        backgroundFileList.append([])
        for f in get_recursive_file_list(background_path+'/'+d):
            backgroundFileList[-1].append(f)
    return backgroundFileList

def custom_randint(min, max):
    min = round(min)
    max = round(max)
    if min == max:
        return min
    return nr.randint(min, max)

def composite(set_name,set_index,
              balanced_background_file_list,
              balanced_object_file_list,
              config,
              label_map,
              img_out_path,
              out_size_w=1600, out_size_h=1200,
              export_crop=False,
              export_arena=True):
    if export_crop and export_arena:
        raise Error('export_crop and export_arena are currently not supported at the same time')
    tf_example = []
    bg_file = balanced_choice(balanced_background_file_list)
    bg = Image.open(bg_file).convert('RGB')
    # TODO: check size of source backgrounds
    bg = bg.resize((out_size_w, out_size_h), resample=PIL.Image.LANCZOS)

    # Don't use jpg for small image sizes: Compression artifacts around copter
    filename = set_name + '-img'+str(set_index)+'.jpg'

    xmins,xmaxs,ymins,ymaxs = [],[],[],[]
    subclasses_text,subclasses,classes_text,classes = [],[],[],[]
    orientations, dist_to_cam, iframe_w, iframe_h = [],[],[],[]
    for j in range(nr.randint(1,4)) if export_arena else range(2):
        obj_meta = balanced_choice(balanced_choice(balanced_choice(balanced_object_file_list)))
        obj_type = obj_meta["robot_type"]
        obj = Image.open(obj_meta['crop'])

        # random scale
        scl_factor = out_size_w/1600
        scl_min = config[obj_type]['scl_min']
        scl_max = config[obj_type]['scl_max']
        if scl_min == scl_max: scl = scl_min
        else: scl = nr.uniform(scl_min, scl_max)
        obj = obj.resize((int(scl*obj.width*scl_factor),int(scl*obj.height*scl_factor)), resample=PIL.Image.LANCZOS)

        # random rotation
        rot = nr.randint(360)
        obj = obj.rotate(rot, resample=Image.BICUBIC, expand=True)
        obj=obj.crop(obj.getbbox())

        # color transformation
        brightness = nr.uniform(config["crop_brightness_augmentation"][0],
                                config["crop_brightness_augmentation"][1])
        enhancer = ImageEnhance.Brightness(obj)
        obj = enhancer.enhance(brightness)

        # random translation
        pos = (nr.randint(0, bg.width-obj.width), nr.randint(0, bg.height-obj.height))

        bg.paste(obj, pos, obj.split()[-1])

        # Prepare meta data
        xmins.append(pos[0]/bg.width)
        xmaxs.append((pos[0]+obj.width)/bg.width)
        ymins.append(pos[1]/bg.height)
        ymaxs.append((pos[1]+obj.height)/bg.height)
        orientations.append(rot)
        subclass = obj_meta['robot_type'] + '_' + obj_meta['identification']
        subclasses_text.append(str.encode(subclass))
        subclasses.append(label_map[subclass])
        classes_text.append(str.encode(obj_meta['robot_type']))
        classes.append(label_map[obj_meta['robot_type']])

        # calculate height
        camera_ground_dist = 310
        if obj_meta['robot_type'] == "copter":
            zero_width = 76
            iframe_w.append(float(obj_meta["inner_frame_width"])*scl)
            iframe_h.append(float(obj_meta["inner_frame_height"])*scl)
            # TODO: check parameters first
            # TODO: does local width makes sense?! (consider robot rotation)
            #dist_to_cam.append(camera_ground_dist * zero_width / iframe_w[-1])
        else:
            dist_to_cam.append(0)

        # Save images
        if export_arena:
            bg.save(img_out_path+filename)
        if export_crop:
            out_var = config['sstage_out_var']
            in_var = config['sstage_in_var']
            obj_w = obj.width
            obj_h = obj.height
            img_crop = bg.crop((
                pos[0]+custom_randint(-out_var*obj_w, +in_var*obj_w),
                pos[1]+custom_randint(-out_var*obj_h, +in_var*obj_h),
                pos[0]+obj_w-custom_randint(-out_var*obj_w, +in_var*obj_w),
                pos[1]+obj_h-custom_randint(-out_var*obj_h, +in_var*obj_h)
            ))
            crop_name = filename.replace('.jpg','-{}.jpg'.format(j))
            img_crop.save(img_out_path+crop_name)
            with tf.gfile.GFile(img_out_path+crop_name, 'rb') as fid:
                encoded_image_data = fid.read()
            tf_example.append(tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(img_crop.height),
                'image/width': dataset_util.int64_feature(img_crop.width),
                'image/filename': dataset_util.bytes_feature(crop_name.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(crop_name.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_image_data),
                'image/format': dataset_util.bytes_feature(b'jpg'),
                'image/object/pose/orientation': dataset_util.float_list_feature([orientations[j]]),
                'image/object/class/text': dataset_util.bytes_list_feature([classes_text[j]]),
                'image/object/class/label': dataset_util.int64_list_feature([classes[j]]),
                'image/object/subclass/text': dataset_util.bytes_list_feature([subclasses_text[j]]),
                'image/object/subclass/label': dataset_util.int64_list_feature([subclasses[j]]),
            })))

    if export_arena:
        with tf.gfile.GFile(img_out_path+filename, 'rb') as fid:
            encoded_image_data = fid.read()
        tf_example.append(tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(bg.height),
            'image/width': dataset_util.int64_feature(bg.width),
            'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature(b'jpg'),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/pose/orientation': dataset_util.float_list_feature(orientations),
            'image/object/pose/iframe_w': dataset_util.float_list_feature(iframe_w),
            'image/object/pose/iframe_h': dataset_util.float_list_feature(iframe_h),
            'image/object/pose/dist_to_cam': dataset_util.float_list_feature(dist_to_cam),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/subclass/text': dataset_util.bytes_list_feature(subclasses_text),
            'image/object/subclass/label': dataset_util.int64_list_feature(subclasses),
        })))
    return tf_example
