import json
import os

def transform_rois(in_dirs, out_dir, default_dir_class, out_name="rois.json"):
        out_file = os.path.join(out_dir, out_name)
        annots = []
        with open(out_file, "w") as o:
            for i, in_dir in enumerate(in_dirs):
                class_file = os.path.join(in_dir, "classes.json")
                annot_file = os.path.join(in_dir, "annotations.json")
                classes = {}
                without_id = 0
                without_points = 0
                if not os.path.exists(class_file):
                    raise FileNotFoundError(f"dir {in_dir} has no classes.json file")
                if not os.path.exists(annot_file):
                    raise FileNotFoundError(f"dir {in_dir} has no annotations.json file")
                with open(class_file, "r") as f:
                    obj = json.load(f)
                    for clazz in obj:
                        classes[clazz['id']] = clazz['name']
                with open(annot_file, "r") as f:
                    obj = json.load(f) 
                    for (img_file, boxes) in obj.items():
                        box_xy = []
                        box_classes = []
                        for b in boxes:
                            clazz = ""
                            if "points" not in b:
                                without_points += 1
                                continue
                            if "classId" not in b:
                                without_id += 1
                                if default_dir_class is not None:
                                    clazz = default_dir_class[i]
                                else:
                                    continue
                            else:
                                clazz = classes[b["classId"]]
                            box_classes.append(clazz)
                            p = b["points"]
                            box_xy.append([p["x1"],p["y1"],p["x2"],p["y2"]])

                        clazz = ""
                        if default_dir_class is not None:
                            clazz = default_dir_class[i]
                        annots.append({
                            "file_name": f"{clazz}/{img_file}",
                            "class": box_classes,
                            "boxes": box_xy
                        })
                print(f"{in_dir}: no id: {without_id}, no points: {without_points}")
            json.dump(annots, o)

if __name__ == "__main__":
    dirs = [
        "/media/lmn/41BA76B7045343B9/dev/school/projectzeamays/data/annot/faw1",
        "/media/lmn/41BA76B7045343B9/dev/school/projectzeamays/data/annot/faw2",
        "/media/lmn/41BA76B7045343B9/dev/school/projectzeamays/data/annot/faw3",
        "/media/lmn/41BA76B7045343B9/dev/school/projectzeamays/data/annot/zinc",
    ]
    dir_classes = ['faw', 'faw', 'faw', 'zinc']
    out_dir = "/media/lmn/41BA76B7045343B9/dev/school/projectzeamays/data/annot"
    transform_rois(dirs, out_dir, dir_classes)



        