import os
import random
import math
import numpy as np
import datetime
from PIL import Image, ImageDraw, ImageSequence
from tqdm import tqdm

# ================= 配置 =================
FOLDER = "images"
CANVAS_SIZE = (4000, 2500)
IMAGE_COUNT = 700
MIN_SIZE, MAX_SIZE = 250, 400
MAX_OVERLAP_IOU = 0.4
MAX_ATTEMPTS = 150
ROTATION_RANGE = (-30, 30)
ANCHOR_DENSITY_FACTOR = 0.009
HEAD_RATIO = 0.45

DEBUG_MODE = True
VISUALIZE_ANCHORS = True if DEBUG_MODE else False

#种子
SEED = None
if SEED is None:
    SEED = random.randint(0, 9999999)
print(f"当前随机种子: {SEED}")
random.seed(SEED)
np.random.seed(SEED)

#工具函数
def has_transparency(img):
    if img.mode != "RGBA":
        return False
    alpha = img.getchannel("A")
    return np.any(np.array(alpha) < 255)

def random_size(min_size, max_size):
    size = int(np.random.lognormal(mean=4.5, sigma=0.35))
    return max(min_size, min(size, max_size))

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    inter_x1, inter_y1 = max(x1, a1), max(y1, b1)
    inter_x2, inter_y2 = min(x2, a2), min(y2, b2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (a2 - a1) * (b2 - b1)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

def head_box(box, ratio=HEAD_RATIO):
    x1, y1, x2, y2 = box
    head_height = int((y2 - y1) * ratio)
    return (x1, y1, x2, y1 + head_height)

def is_head_blocked(new_box, placed_boxes, ratio=HEAD_RATIO):
    for other in placed_boxes.all_boxes():
        hx1, hy1, hx2, hy2 = head_box(other, ratio)
        nx1, ny1, nx2, ny2 = new_box
        if nx1 <= hx1 and ny1 <= hy1 and nx2 >= hx2 and ny2 >= hy2:
            return True
    return False

#缓存加载
def load_images(folder):
    cache = {}
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]
    total = len(files)
    if total == 0:
        raise ValueError("文件夹里没有图片！")

    for idx, file in enumerate(files, 1):
        path = os.path.join(folder, file)
        img = Image.open(path).convert("RGBA")
        cache[path] = img
        if DEBUG_MODE:
            print(f"[缓存] {idx}/{total} 加载: {file} (大小: {img.size})")

    if DEBUG_MODE:
        print(f"[缓存] 总图片数量: {len(cache)}")
    return cache

# ================== 放置盒子管理 ==================
class PlacedBoxes:
    def __init__(self, cell_size=200):
        self.grid = {}
        self.cell_size = cell_size

    def _grid_coords(self, box):
        x1, y1, x2, y2 = box
        return (x1 // self.cell_size, y1 // self.cell_size,
                x2 // self.cell_size, y2 // self.cell_size)

    def add(self, box):
        gx1, gy1, gx2, gy2 = self._grid_coords(box)
        for gx in range(gx1, gx2 + 1):
            for gy in range(gy1, gy2 + 1):
                self.grid.setdefault((gx, gy), []).append(box)

    def overlap(self, box, max_iou):
        gx1, gy1, gx2, gy2 = self._grid_coords(box)
        for gx in range(gx1, gx2 + 1):
            for gy in range(gy1, gy2 + 1):
                if (gx, gy) in self.grid:
                    for other in self.grid[(gx, gy)]:
                        if iou(box, other) > max_iou:
                            return True
        return False

    def all_boxes(self):
        for v in self.grid.values():
            for b in v:
                yield b

# ================== 锚点生成 ==================
class LazyAnchors:
    def __init__(self, canvas_size, density):
        self.w, self.h = canvas_size
        self.cell_size = 200
        self.count = int(self.w * self.h * density)
        self.anchors = [(random.randint(0, self.w), random.randint(0, self.h)) for _ in range(self.count)]
        if DEBUG_MODE:
            x_cells = math.ceil(self.w / self.cell_size)
            y_cells = math.ceil(self.h / self.cell_size)
            total_cells = x_cells * y_cells
            print(f"[锚点信息] 锚点数量: {self.count}, 分区数量: {total_cells}, 分区大小: {self.cell_size}x{self.cell_size}")

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.count:
            raise StopIteration
        val = self.anchors[self.idx]
        self.idx += 1
        return val

# ================== 图片放置 ==================
def place_images(canvas, cache, image_count, visualize_anchors=False):
    opaque_files = [f for f, img in cache.items() if not has_transparency(img)]
    gif_files = [f for f in cache if f.lower().endswith(".gif")]
    base_layer_files = list(set(opaque_files + gif_files))
    transparent_files = [f for f, img in cache.items() if has_transparency(img) and f not in base_layer_files]

    total_files = len(base_layer_files) + len(transparent_files)
    if total_files < image_count and len(transparent_files) > 0:
        needed = image_count - total_files
        supplement = random.choices(transparent_files, k=needed)
        transparent_files += supplement

    file_layers = [base_layer_files, transparent_files]

    placed = PlacedBoxes()
    anchors = LazyAnchors(CANVAS_SIZE, ANCHOR_DENSITY_FACTOR)
    anchors_iter = iter(anchors)

    if visualize_anchors:
        vis = canvas.copy()
        draw = ImageDraw.Draw(vis)
        for x, y in anchors.anchors:
            draw.ellipse((x-3, y-3, x+3, y+3), fill=(255,0,0,255))
        vis.save("anchors_debug.png")
        print("锚点可视化已保存：anchors_debug.png")

    used = 0
    failed_attempts = 0
    consecutive_failures = 0

    for layer_idx, files in enumerate(file_layers):
        for f in tqdm(files, desc=f"放置 {'底层' if layer_idx==0 else '上层'} 图片"):
            if used >= image_count or consecutive_failures >= 10:
                if consecutive_failures >= 10:
                    print("连续10次放置失败，提前结束放置")
                break
            img = cache[f]
            for attempt in range(1, MAX_ATTEMPTS+1):
                size = random_size(MIN_SIZE, MAX_SIZE)
                img_resized = img.resize((size, int(size * img.height / img.width)))
                angle = random.uniform(*ROTATION_RANGE)
                rotated = img_resized.rotate(angle, expand=True)

                try:
                    x, y = next(anchors_iter)
                except StopIteration:
                    anchors_iter = iter(LazyAnchors(CANVAS_SIZE, ANCHOR_DENSITY_FACTOR))
                    x, y = next(anchors_iter)

                box = (x, y, x + rotated.width, y + rotated.height)

                if (box[2] <= CANVAS_SIZE[0] and box[3] <= CANVAS_SIZE[1]
                    and not placed.overlap(box, MAX_OVERLAP_IOU)
                    and not is_head_blocked(box, placed)):
                    canvas.paste(rotated, (x, y), rotated)
                    placed.add(box)
                    used += 1
                    consecutive_failures = 0  # 放置成功，重置
                    if DEBUG_MODE:
                        print(f"[放置尝试] {f} 尝试次数: {attempt}")
                    break
            else:
                failed_attempts += 1
                consecutive_failures += 1
                if DEBUG_MODE:
                    print(f"[放置失败] {f} 未能放置")
        if consecutive_failures >= 10:
            break

    if DEBUG_MODE:
        print(f"[放置统计] 总放置成功: {used}, 总失败次数: {failed_attempts}")

    return canvas

# ================== 主函数 ==================
def main():
    cache = load_images(FOLDER)
    canvas = Image.new("RGBA", CANVAS_SIZE, (255, 255, 255, 0))
    result = place_images(canvas, cache, IMAGE_COUNT, visualize_anchors=VISUALIZE_ANCHORS)
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = f"suzume_{now_str}.png"
    canvas.save(output, "PNG")
    print(f"好耶！已保存 {output}")

if __name__ == "__main__":
    main()