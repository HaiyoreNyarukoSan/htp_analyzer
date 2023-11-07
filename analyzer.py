from pathlib import Path
from typing import Tuple, List, Union

import ultralytics.engine.results
from ultralytics import YOLO

from evaluator import ModelEvaluator
from utils import find_static_file, fetch_pickle

BoundingBoxType = ultralytics.engine.results.Boxes
YOLOResultType = ultralytics.engine.results.Results
MAX_SCORE = 10

DATA_CATEGORY = ['나무', '남자사람', '여자사람', '집']

agreeableness = '우호성'
conscientiousness = '성실성'
extraversion = '외향성'
neuroticism = '신경성'
openness_to_experience = '경험에 대한 개방성'

STAT_TYPE = (agreeableness, conscientiousness, extraversion, neuroticism, openness_to_experience)
stat_deserializer = lambda scores: dict((t, s) for t, s in zip(STAT_TYPE, scores))


# Create your views here.
def yolo_model_fetcher(category):
    model_path = f'model/checkpoint_{category}.pt'
    fullpath = find_static_file(model_path)
    return YOLO(fullpath)


models = dict((category, yolo_model_fetcher(category)) for category in DATA_CATEGORY)
IMAGE_SIZE = 1280
image_area = IMAGE_SIZE * IMAGE_SIZE


def get_bounding_boxes_single_category(model: YOLO, images: List[Path]):
    result_list: List[YOLOResultType] = []
    i, batch_size = 0, 32
    while sub_images := images[i:i + batch_size]:
        result_list.extend(model.predict(sub_images))
        i += batch_size
    return list(result.boxes for result in result_list)


def get_bounding_boxes(containers: Tuple[List[Path]]):
    # containers : ([t1,t2],[m1,m2],[w1,w2],[h1,h2])
    assert (len(containers) == len(DATA_CATEGORY))
    boxes_list_categorized = []
    for category, images in zip(DATA_CATEGORY, containers):
        model: YOLO = models[category]
        if not (images and model): continue
        cboxes: List[BoundingBoxType] = get_bounding_boxes_single_category(model, images)
        boxes_list_categorized.append(cboxes)
    return tuple(boxes_list_categorized)


def analyzer(images_list: Union[Tuple[Path], List[Tuple[Path]]]):
    # images_list : (tree,man,woman,house) or [(t1,m1,w1,h1),(t2,m2,w2,h2)]
    if isinstance(images_list[0], Path):
        images_list: List[Tuple[Path]] = [images_list]
    containers: Tuple[List[Path]] = tuple(list(r) for r in zip(*images_list))
    boxes_list_categorized: Tuple[List[BoundingBoxType]] = get_bounding_boxes(containers)
    total_score = ModelEvaluator.stat_evaluater(boxes_list_categorized)
    return [stat_deserializer(score) for score in total_score]
