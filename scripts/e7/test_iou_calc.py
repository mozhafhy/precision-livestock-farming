import json
from pathlib import Path
from collections import defaultdict

def poly_to_mask(poly_coords, width=224, height=224):
    mask = [[0]*width for _ in range(height)]
    pts = [(poly_coords[i], poly_coords[i+1]) for i in range(0, len(poly_coords), 2)]
    n = len(pts)
    if n < 3:
        return mask
    min_y = max(0, int(min(p[1] for p in pts)))
    max_y = min(height - 1, int(max(p[1] for p in pts)))
    for y in range(min_y, max_y + 1):
        scan_y = y + 0.5
        nodes = []
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            if (p1[1] < scan_y and p2[1] >= scan_y) or (p2[1] < scan_y and p1[1] >= scan_y):
                x = p1[0] + (scan_y - p1[1]) / (p2[1] - p1[1]) * (p2[0] - p1[0])
                nodes.append(x)
        nodes.sort()
        for i in range(0, len(nodes) - 1, 2):
            x_start = max(0, int(nodes[i]))
            x_end = min(width - 1, int(nodes[i+1]))
            for x in range(x_start, x_end + 1):
                mask[y][x] = 1
    return mask

p_mhs = Path(r"D:\Richard\Semester 5\precision-livestock-farming\dataset\Komparasi perbaikan ahli E7\Hasil Kerja Mahasiswa\train\_annotations.coco.json")
p_ahli = Path(r"D:\Richard\Semester 5\precision-livestock-farming\dataset\Komparasi perbaikan ahli E7\Perbaikan Ahli\train\_annotations.coco.json")

with open(p_mhs) as f: mhs = json.load(f)
with open(p_ahli) as f: ahli = json.load(f)

mhs_ann = defaultdict(list)
for a in mhs['annotations']: mhs_ann[a['image_id']].append(a)
ahli_ann = defaultdict(list)
for a in ahli['annotations']: ahli_ann[a['image_id']].append(a)

ious = []
for img in mhs['images']:
    img_id = img['id']
    w, h = img['width'], img['height']
    mask_m = [[0]*w for _ in range(h)]
    for a in mhs_ann[img_id]:
        for seg in a['segmentation']:
            sub = poly_to_mask(seg, w, h)
            for y in range(h):
                for x in range(w):
                    if sub[y][x]: mask_m[y][x] = 1
    mask_a = [[0]*w for _ in range(h)]
    for a in ahli_ann[img_id]:
        for seg in a['segmentation']:
            sub = poly_to_mask(seg, w, h)
            for y in range(h):
                for x in range(w):
                    if sub[y][x]: mask_a[y][x] = 1
    inter = sum(mask_m[y][x] and mask_a[y][x] for y in range(h) for x in range(w))
    union = sum(mask_m[y][x] or mask_a[y][x] for y in range(h) for x in range(w))
    iou = inter / union if union > 0 else 1.0
    ious.append(iou)

mean_iou = sum(ious) / len(ious) * 100
print(f"Mean IoU (30 images): {mean_iou:.2f}%")
print(f"Min IoU: {min(ious)*100:.2f}%")
print(f"Max IoU: {max(ious)*100:.2f}%")
