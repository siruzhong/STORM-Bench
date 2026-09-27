"""Draw visible target masks without revealing occluded geometry."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def target_mask(segmentation, color):
    value = np.asarray(color, dtype=float)
    if value.max() <= 1:
        value *= 255
    return np.all(np.abs(segmentation.astype(np.int16) - np.rint(value).astype(np.int16)) <= 1, axis=-1)


def highlight(rgb, mask, caption, action, timestamp):
    result = rgb.copy()
    pixels = int(mask.sum())
    if pixels:
        result[mask] = (0.55 * result[mask] + 0.45 * np.array([255, 220, 25])).astype(np.uint8)
        binary = Image.fromarray(mask.astype(np.uint8) * 255)
        expanded = np.array(binary.filter(ImageFilter.MaxFilter(5))) > 0
        outline = expanded & ~mask
        result[outline] = [255, 230, 30]
    header_height = 64
    image = Image.new('RGB', (result.shape[1], result.shape[0] + header_height), (18, 22, 28))
    image.paste(Image.fromarray(result), (0, header_height))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    state = f'{pixels} visible pixels' if pixels else 'NOT VISIBLE'
    draw.text((10, 7), f'{timestamp:06.2f}s  {caption}  |  {state}', fill=(255, 230, 30), font=font)
    draw.text((10, 33), action[:85], fill=(240, 240, 240), font=ImageFont.load_default(size=13))
    if pixels:
        yy, xx = np.where(mask)
        box = (max(0, int(xx.min()) - 7), header_height + max(0, int(yy.min()) - 7),
               min(image.width - 1, int(xx.max()) + 7), min(image.height - 1, header_height + int(yy.max()) + 7))
        if box[3] >= box[1]:
            draw.rectangle(box, outline=(255, 230, 30), width=2)
    return np.array(image)


def camera_audit(episode):
    path = Path(episode) / 'logs/camera_collision.csv'
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(int(row['overlaps']) for row in rows):
        raise RuntimeError('Camera collision audit failed; inspect camera_collision.csv')
    return {'camera_samples': len(rows),
            'camera_retracted_samples': sum(int(row['blocked']) for row in rows),
            'camera_overlap_samples': sum(int(row['overlaps']) for row in rows)}
