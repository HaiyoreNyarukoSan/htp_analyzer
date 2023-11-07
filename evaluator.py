from pathlib import Path
from typing import Tuple, List

import torch
import ultralytics
from torch import nn

from utils import xyxys_in_xyxys, find_static_file, fetch_pickle

BoundingBoxType = ultralytics.engine.results.Boxes
InputDataType = Tuple[List[BoundingBoxType]]

LABELS = {
    '나무': ['나무전체', '기둥', '수관', '가지', '뿌리', '나뭇잎', '꽃', '열매', '그네', '새', '다람쥐', '구름', '달', '별'],
    '남자사람': ['사람전체', '머리', '얼굴', '눈', '코', '입', '귀', '머리카락', '목', '상체', '팔', '손', '다리', '발', '단추', '주머니', '운동화',
             '남자구두'],
    '여자사람': ['사람전체', '머리', '얼굴', '눈', '코', '입', '귀', '머리카락', '목', '상체', '팔', '손', '다리', '발', '단추', '주머니', '운동화',
             '여자구두'],
    '집': ['집전체', '지붕', '집벽', '문', '창문', '굴뚝', '연기', '울타리', '길', '연못', '산', '나무', '꽃', '잔디', '태양']
}

agreeableness = '우호성'
conscientiousness = '성실성'
extraversion = '외향성'
neuroticism = '신경성'
openness_to_experience = '경험에 대한 개방성'

STAT_TYPE = (agreeableness, conscientiousness, extraversion, neuroticism, openness_to_experience)


class BoxData:
    def __init__(self, boxes: List[BoundingBoxType], labels: List[str], target: str):
        if isinstance(target, str):
            index = labels.index(target)
            self._boxes = [box for box in boxes if int(box.cls) == index]
        else:
            indices = [labels.index(t) for t in target if t in labels]
            self._boxes = [box for box in boxes if int(box.cls) in indices]
        self._count = len(self._boxes)

    @property
    def _xyxys(self):
        return map(lambda box: tuple(map(int, box.xyxy[0])), self._boxes)

    @property
    def _whs(self):
        return map(lambda box: tuple(map(int, box.xywh[0]))[2:], self._boxes)

    @property
    def count(self):
        return self._count

    @property
    def xyxys(self):
        return self._xyxys

    @property
    def whs(self):
        return self._whs

    @property
    def area_sum(self):
        return sum(wh[0] * wh[1] for wh in self._whs)

    @property
    def xmin_ave(self):
        return sum(xyxy[0] for xyxy in self._xyxys) / self._count if self._count else 0

    @property
    def ymin_ave(self):
        return sum(xyxy[1] for xyxy in self._xyxys) / self._count if self._count else 0

    @property
    def xmax_ave(self):
        return sum(xyxy[2] for xyxy in self._xyxys) / self._count if self._count else 0

    @property
    def ymax_ave(self):
        return sum(xyxy[3] for xyxy in self._xyxys) / self._count if self._count else 0

    @property
    def w_sum(self):
        return sum(wh[0] for wh in self._whs)

    @property
    def h_sum(self):
        return sum(wh[1] for wh in self._whs)

    @property
    def wh_sum(self):
        return sum(sum(wh) for wh in self._whs)

    def count_of_is_inside(self, boxdata):
        return xyxys_in_xyxys(self._xyxys, boxdata.xyxys, (('xy', 'in'),)) / self.count if self.count else 0

    def count_of_is_above(self, boxdata):
        return xyxys_in_xyxys(self._xyxys, boxdata.xyxys, (('x', 'in'), ('y', 'lt'))) / self.count if self.count else 0


def get_args(boxes_list: Tuple[BoundingBoxType]):
    tree_boxes, man_boxes, woman_boxes, house_boxes = boxes_list
    # 나무
    boxes, labels = tree_boxes, LABELS['나무']
    tree_branchs = BoxData(boxes, labels, '가지')
    tree_clouds = BoxData(boxes, labels, '구름')
    tree_fruits = BoxData(boxes, labels, '열매')
    tree_leaves = BoxData(boxes, labels, '나뭇잎')
    tree_roots = BoxData(boxes, labels, '뿌리')
    tree_trunks = BoxData(boxes, labels, '기둥')
    # 남자사람
    boxes, labels = man_boxes, LABELS['남자사람']
    man_arms = BoxData(boxes, labels, '팔')
    man_bodys = BoxData(boxes, labels, '사람전체')
    man_chests = BoxData(boxes, labels, '상체')
    man_eyes = BoxData(boxes, labels, '눈')
    man_faces = BoxData(boxes, labels, '얼굴')
    man_foots = BoxData(boxes, labels, '발')
    man_hands = BoxData(boxes, labels, '손')
    man_legs = BoxData(boxes, labels, '다리')
    man_necks = BoxData(boxes, labels, '목')
    man_noses = BoxData(boxes, labels, '코')
    man_shoes = BoxData(boxes, labels, ('운동화', '남자구두', '여자구두'))
    # 여자사람
    boxes, labels = woman_boxes, LABELS['여자사람']
    woman_arms = BoxData(boxes, labels, '팔')
    woman_bodys = BoxData(boxes, labels, '사람전체')
    woman_chests = BoxData(boxes, labels, '상체')
    woman_eyes = BoxData(boxes, labels, '눈')
    woman_faces = BoxData(boxes, labels, '얼굴')
    woman_foots = BoxData(boxes, labels, '발')
    woman_hands = BoxData(boxes, labels, '손')
    woman_legs = BoxData(boxes, labels, '다리')
    woman_necks = BoxData(boxes, labels, '목')
    woman_noses = BoxData(boxes, labels, '코')
    woman_shoes = BoxData(boxes, labels, ('운동화', '남자구두', '여자구두'))
    # 집
    boxes, labels = house_boxes, LABELS['집']
    house_doors = BoxData(boxes, labels, '문')
    house_flowers = BoxData(boxes, labels, '꽃')
    house_grasses = BoxData(boxes, labels, '잔디')
    house_houses = BoxData(boxes, labels, '집전체')
    house_mountains = BoxData(boxes, labels, '산')
    house_roads = BoxData(boxes, labels, '길')
    house_roofs = BoxData(boxes, labels, '지붕')
    house_smokes = BoxData(boxes, labels, '연기')
    house_suns = BoxData(boxes, labels, '태양')
    house_trees = BoxData(boxes, labels, '나무')
    house_windows = BoxData(boxes, labels, '창문')
    # 모델 입력 반환
    return (tree_branchs.area_sum,
            tree_branchs.count,
            tree_branchs.wh_sum,
            tree_clouds.count,
            tree_fruits.ymax_ave,
            tree_leaves.count,
            tree_leaves.wh_sum,
            tree_roots.area_sum,
            tree_roots.count,
            tree_trunks.area_sum,
            tree_trunks.h_sum,
            tree_trunks.ymax_ave,
            man_arms.wh_sum,
            man_bodys.area_sum,
            man_bodys.h_sum,
            man_bodys.w_sum,
            man_bodys.wh_sum,
            man_chests.area_sum,
            man_chests.h_sum,
            man_chests.w_sum,
            man_eyes.area_sum,
            man_faces.area_sum,
            man_foots.area_sum,
            man_foots.count,
            man_hands.area_sum,
            man_hands.count,
            man_legs.area_sum,
            man_legs.wh_sum,
            man_necks.area_sum,
            man_noses.area_sum,
            man_shoes.area_sum,
            woman_arms.wh_sum,
            woman_bodys.area_sum,
            woman_bodys.h_sum,
            woman_bodys.w_sum,
            woman_bodys.wh_sum,
            woman_chests.area_sum,
            woman_chests.h_sum,
            woman_chests.w_sum,
            woman_eyes.area_sum,
            woman_faces.area_sum,
            woman_foots.area_sum,
            woman_foots.count,
            woman_hands.area_sum,
            woman_hands.count,
            woman_legs.area_sum,
            woman_legs.wh_sum,
            woman_necks.area_sum,
            woman_noses.area_sum,
            woman_shoes.area_sum,
            house_doors.area_sum,
            house_doors.count,
            house_flowers.count,
            house_grasses.count,
            house_houses.area_sum,
            house_mountains.count,
            house_roads.count,
            house_roofs.area_sum,
            house_smokes.count,
            house_suns.count,
            house_trees.count,
            house_windows.count,
            house_windows.count_of_is_inside(house_roofs))


# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.activ1 = nn.Softplus()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.activ2 = nn.Softplus()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.activ3 = nn.Softplus()
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.activ4 = nn.Softplus()
        self.fclast = nn.Linear(hidden_size4, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activ1(out)
        out = self.fc2(out)
        out = self.activ2(out)
        out = self.fc3(out)
        out = self.activ3(out)
        out = self.fc4(out)
        out = self.activ4(out) / 10
        out = self.fclast(out)
        out = 10 * self.sigmoid(out)
        return out


def tensor_to_stat(tensor: torch.tensor):
    return [int(v) for v in torch.round(tensor)]


class ModelEvaluator:
    @staticmethod
    def stat_evaluater(boxes_list_categorized: InputDataType):
        input_size = 63  # Number of input features
        hidden_size1 = 48  # Number of neurons in the first hidden layer
        hidden_size2 = 32  # Number of neurons in the second hidden layer
        hidden_size3 = 32  # Number of neurons in the third hidden layer
        hidden_size4 = 16  # Number of neurons in the third hidden layer
        output_size = 5  # Number of output neurons

        args_list = [get_args(boxes_list) for boxes_list in zip(*boxes_list_categorized)]

        normalizer_path = Path('model/normalizer.pkl')
        model_path = Path('model/stat_evaluator.pt')

        (mean, std) = fetch_pickle(find_static_file(normalizer_path))
        model = NeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size)
        model.load_state_dict(torch.load(find_static_file(model_path)))

        args_list = (args_list - mean) / std

        res = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(args_list), batch_size):
                data = torch.tensor(args_list[i:i + batch_size], dtype=torch.float32)
                res.extend(model(data))
        return [tensor_to_stat(tensor) for tensor in res]
