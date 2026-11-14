import base64
import io

import pytest
from PIL import Image

import server


def image_data_url():
    output = io.BytesIO()
    Image.new('RGB', (8, 8), (240, 240, 240)).save(output, 'JPEG')
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode()


def test_validate_project_rejects_mixed_sources():
    project = {
        "version": 1,
        "names": ["红色", "黄色", "蓝色"],
        "samples": [
            {"label": 0, "source": "camera", "image": image_data_url()},
            {"label": 1, "source": "simulation", "image": image_data_url()},
        ],
    }
    with pytest.raises(ValueError):
        server.validate_project(project)


def test_decode_image_rejects_invalid_payload():
    with pytest.raises(ValueError):
        server.decode_image("data:image/jpeg;base64," + base64.b64encode(b"invalid").decode())

