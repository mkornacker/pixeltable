import numpy as np
import PIL

from pixeltable.type_system import ImageType, ArrayType, FloatType
from pixeltable.function import Function, FunctionRegistry
from pixeltable.exceptions import Error

def _draw_boxes(img: PIL.Image.Image, boxes: np.ndarray) -> PIL.Image.Image:
    if len(boxes.shape) != 2 or boxes.shape[1] != 4:
        raise Error(f'draw(): boxes needs to have shape (None, 4) but instead has shape {boxes.shape}')
    result = img.copy()
    d = PIL.ImageDraw.Draw(result)
    for i in range(boxes.shape[0]):
        d.rectangle(list(boxes[i]), width=3)
    return result

draw_boxes = Function.make_library_function(
    ImageType(), [ImageType(), ArrayType((None, 4), dtype=FloatType())], __name__, '_draw_boxes')
FunctionRegistry.get().register_function(__name__, 'draw_boxes', draw_boxes)

__all__ = [
    draw_boxes,
]