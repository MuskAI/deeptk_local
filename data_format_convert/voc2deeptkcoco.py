"""
created by haoran
time:2021-1-2
base on :https://github.com/Tony607/voc2coco
new function:
1. 检查
2. 纠正小错误
"""
import sys
import os
import json
import xml.etree.ElementTree as ET
import glob
import argparse
import re

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = None
# FILE_NAME_KEYWORDS = 'ship'
#If necessary, pre-define category and its id，
PRE_DEFINE_CATEGORIES = {'fire':1}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        # filename的后缀数字将作为id
        filename = re.findall("\d+", filename)[0]

        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.

    Arguments:
        xml_files {list} -- A list of xml file paths.

    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_files, json_file,image_dir=None):

    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    for idx,xml_file in enumerate(xml_files):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")

        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        ## The filename must be a number;
        # TODO 这里会将文件名作为id，需要先把名称转化为数字
        # image_id = get_filename_as_int(filename)
        image_id = idx+1
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)

        # file_name judge

        # if FILE_NAME_KEYWORDS not in file_name:
        #     file_name = FILE_NAME_KEYWORDS+file_name


        if os.path.exists(os.path.join(image_dir,filename)):
            pass
        else:
            # 当不匹配的时候采取补救措施
            filename = os.path.basename(xml_file)

            # 可能的图像格式
            _suffix = ('.jpg','.JPG','png','PNG','.JPEG','.jpeg')
            _match_flag = False
            for _i in _suffix:
                _path = filename.split('.')[0]+_i
                if os.path.exists(os.path.join(image_dir,_path)):
                    _match_flag = True
                    filename = _path
                    break
                else:
                    pass


            if _match_flag is False:
                print('图片{}不匹配'.format(filename))
                continue

        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        ## Currently we do not support segmentation.
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, "object"):
            none_c = False
            category = get_and_check(obj, "name", 1).text
            # if category == 'none':
            #     category = 'otherboat'
            #     none_c = True

            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
            ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)


            ann = {
                "area": o_width * o_height,
                "iscrowd": 1 if none_c else 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC annotation to COCO format."
    )
    parser.add_argument("--xml_dir", help="Directory path to xml files.", type=str,default='/Users/musk/Desktop/实习生工作/火焰检测项目/fire-dataset2-train/label')
    parser.add_argument("--json_file",help="Output COCO format json file.", type=str,default='../tmp/fire_dataset2_train.json')
    parser.add_argument("--image_dir", help="check_image", type=str,default='/Users/musk/Desktop/实习生工作/火焰检测项目/fire-dataset2-train/image')

    args = parser.parse_args()
    xml_files = glob.glob(os.path.join(args.xml_dir, "*.xml"))
    dirty_file = [d for d in xml_files if '.xml' not in d]
    print(dirty_file)
    for d in dirty_file:
        xml_files.remove(d)

    # If you want to do train/test split, you can pass a subset of xml files to convert function.
    print("Number of xml files: {}".format(len(xml_files)))
    convert(xml_files, args.json_file,args.image_dir)
    print("Success: {}".format(args.json_file))
