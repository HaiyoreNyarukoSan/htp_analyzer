from pathlib import Path
from typing import List, Tuple, Dict, Union

from flask import Flask, request, jsonify

from analyzer import analyzer, DATA_CATEGORY

app = Flask(__name__)
# DATA_CATEGORY = ['나무', '남자사람', '여자사람', '집']
categories = ('tree', 'man', 'woman', 'house')
images_suffixes = ('.jpg', 'jpeg', 'webp')


def fetch_files(path: Path, suffixes: Tuple[str]):
    return [f for f in path.iterdir() if f.suffix.lower() in suffixes]


def process_datas(container_path: Path, datas: List[Dict[str, Union[int, str]]]):
    attach_container = lambda container, paths: tuple(container / path for path in paths)
    extract_category_urls = lambda data: tuple(data[category] for category in categories)
    data_ids = [data['id'] for data in datas]
    images_list = [attach_container(container_path, extract_category_urls(data)) for data in datas]
    return data_ids, images_list


@app.post("/evaluate")
def evaluate():
    try:
        data = request.get_json()
        container_url = data.get('container_url')
        datas = data.get('datas', [])

        if not container_url:
            return jsonify({'error': 'URL for images container not provided'}), 400
        if not datas:
            return jsonify({'error': 'No datas provided'}), 400

        container_path = Path(container_url)
        data_ids, images_list = process_datas(container_path, datas)
        scores_list = analyzer(images_list)
        for data_id, scores in zip(data_ids, scores_list):
            scores['id'] = data_id
        return jsonify(scores_list)

    except Exception as e:
        return jsonify({'error': 'An error occurred', 'details': str(e)}), 500
