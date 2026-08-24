"""
FLIP 답안 인식용 YOLO-det 학습 데이터 합성기.

단일-문자 손글씨 데이터셋(xainano 등, 클래스별 폴더)에서 글자를 꺼내
캔버스에 왼->오로 붙여 "답안 문자열" 이미지를 만들고,
붙인 위치가 곧 정답 박스라서 YOLO 라벨을 0초에 자동 생성한다.

실행:
  python synth_data.py --selftest                # 데이터셋 없이 로직 검증
  python synth_data.py --dataset <xainano_dir> --out data --n 4000

xainano 폴더명이 아래 CLASS_FOLDERS와 다르면 그 dict만 고치면 됨.
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# 우리가 쓸 클래스 (id = 인덱스). 정답에 나오는 것만.
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", "/", "=", "pi"]

# 우리 클래스 -> xainano 하위 폴더명.  다운로드한 폴더 보고 안 맞으면 여기만 수정.
CLASS_FOLDERS = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "-": "-",              # 음수/마이너스
    "/": "forward_slash",  # 분수 슬래시 (xainano에서 div 로 되어 있으면 "div" 로 변경)
    "=": "=",
    "pi": "pi",
}

CANVAS_H = 64          # 캔버스 높이(px)
GLYPH_H_RANGE = (30, 48)  # 글자 높이 랜덤 범위
# 글자 간격(px). 샘플마다 한 모드를 골라 그 답 전체에 일관 적용
# (사람은 한 답을 대체로 같은 간격으로 씀). 둘 다 학습해야 함.
GAP_TIGHT = (-12, -2)   # 붙여쓰기/겹침 (예: 붙은 -3, 2/5)
GAP_SPACED = (2, 14)    # 띄어쓰기/단독 (예: 벌어진 365)
PAD = 8                # 좌우/상하 여백
INK_THRESH = 40        # alpha>이 값이면 잉크로 간주 (박스 계산용)


def normalize_glyph(img: Image.Image) -> np.ndarray:
    """어떤 극성이든 '흰 배경 위 검은 잉크'로 통일해 alpha(잉크 강도) 배열 반환."""
    a = np.asarray(img.convert("L"), dtype=np.float32)
    if a.mean() < 127:          # 검은 배경 위 흰 글씨면 반전
        a = 255.0 - a
    ink = 255.0 - a             # 검을수록 잉크 강함
    ink[ink < 0] = 0
    return ink                  # HxW, 0..255


def ink_bbox(ink: np.ndarray):
    """잉크 영역의 타이트 박스 (x1,y1,x2,y2). 없으면 None."""
    ys, xs = np.where(ink > INK_THRESH)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def load_pool(dataset_dir: str):
    """클래스 -> 이미지 파일 경로 리스트."""
    pool = {}
    for cls in CLASSES:
        folder = Path(dataset_dir) / CLASS_FOLDERS[cls]
        if not folder.is_dir():
            raise FileNotFoundError(
                f"클래스 '{cls}' 폴더 없음: {folder}  -> CLASS_FOLDERS 확인/수정"
            )
        files = sorted(p for p in folder.iterdir()   # sorted: run마다 순서 고정(재현성)
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
        if not files:
            raise FileNotFoundError(f"{folder} 에 이미지가 없음")
        pool[cls] = files
    return pool


def fake_pool(tmp: Path):
    """셀프테스트용: 폰트로 글리프를 그려 가짜 데이터셋 생성(데이터셋 불필요)."""
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    glyph_text = {**{c: c for c in "0123456789"}, "-": "-", "/": "/", "=": "=", "pi": "n"}
    pool = {}
    for cls in CLASSES:
        d = tmp / CLASS_FOLDERS[cls]
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(6):
            im = Image.new("L", (45, 45), 255)
            ImageDraw.Draw(im).text((12, 2), glyph_text[cls], fill=0, font=font)
            p = d / f"{i}.png"
            im.save(p)
            paths.append(p)
        pool[cls] = paths
    return pool


def compose_sequence():
    """그럴듯한 답 시퀀스(클래스 라벨 리스트). 모든 클래스가 골고루 나오게."""
    r = random.random()
    if r < 0.45:                                   # 정수 (부호 가능)
        seq = random.choices("0123456789", k=random.randint(1, 3))
        if random.random() < 0.3:
            seq = ["-"] + seq
    elif r < 0.75:                                 # 분수
        num = random.choices("0123456789", k=random.randint(1, 2))
        den = random.choices("0123456789", k=random.randint(1, 2))
        seq = num + ["/"] + den
        if random.random() < 0.3:
            seq = ["-"] + seq
    elif r < 0.9:                                  # pi 포함
        seq = random.choices("0123456789", k=random.randint(0, 2)) + ["pi"]
        if random.random() < 0.4:
            seq += ["/"] + random.choices("0123456789", k=1)
    else:                                          # '=' 노출용 (희소)
        seq = random.choices("0123456789", k=1) + ["="] + \
              random.choices("0123456789", k=random.randint(1, 2))
    return seq


def render_sample(pool):
    """한 장 합성. return (PIL RGB image, [(class_id, x1,y1,x2,y2), ...])."""
    seq = compose_sequence()
    glyphs = []
    for cls in seq:
        ink = normalize_glyph(Image.open(random.choice(pool[cls])))
        h = random.randint(*GLYPH_H_RANGE)
        w = max(1, int(ink.shape[1] * h / ink.shape[0]))
        g = np.asarray(Image.fromarray(ink).resize((w, h)), dtype=np.float32)
        ang = random.uniform(-12, 12)
        g = np.asarray(Image.fromarray(g).rotate(ang, expand=True, resample=Image.BILINEAR),
                       dtype=np.float32)
        glyphs.append((CLASSES.index(cls), g))

    gap = GAP_TIGHT if random.random() < 0.5 else GAP_SPACED  # 이 답의 간격 모드
    total_w = PAD * 2 + sum(g.shape[1] for _, g in glyphs) + \
        gap[1] * max(0, len(glyphs) - 1)
    canvas = np.zeros((CANVAS_H, total_w), dtype=np.float32)  # 잉크 누적(검을수록 큼)
    boxes = []
    x = PAD
    for cid, g in glyphs:
        gh, gw = g.shape
        y = random.randint(PAD, max(PAD, CANVAS_H - gh - PAD))
        # 캔버스 밖으로 삐져나가면 잘라서 붙임(회전으로 커진 경우 대비)
        yh, xw = min(gh, CANVAS_H - y), min(gw, canvas.shape[1] - x)
        sub = g[:yh, :xw]
        canvas[y:y + yh, x:x + xw] = np.maximum(canvas[y:y + yh, x:x + xw], sub)
        bb = ink_bbox(sub)  # 실제로 붙은 부분에서 박스 계산
        if bb:  # 캔버스 좌표로 이동
            bx1, by1, bx2, by2 = bb
            boxes.append((cid, x + bx1, y + by1, x + bx2, y + by2))
        x += gw + random.randint(*gap)

    img = Image.fromarray(255.0 - canvas).convert("RGB")  # 흰 배경 위 검은 잉크
    # 촬영처럼 가벼운 노이즈/블러/명암
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.8)))
    arr = np.asarray(img, dtype=np.float32)
    arr += np.random.normal(0, random.uniform(2, 9), arr.shape)
    arr = np.clip(arr * random.uniform(0.85, 1.0) + random.uniform(0, 20), 0, 255)
    return Image.fromarray(arr.astype(np.uint8)), boxes


def to_yolo(box, W, H):
    _, x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
    bw, bh = (x2 - x1) / W, (y2 - y1) / H
    return box[0], cx, cy, bw, bh


def write_split(pool, out: Path, n, split):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img, boxes = render_sample(pool)
        if not boxes:
            continue
        W, H = img.size
        img.save(out / "images" / split / f"{i:06d}.jpg", quality=90)
        lines = []
        for b in boxes:
            cid, cx, cy, bw, bh = to_yolo(b, W, H)
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / "labels" / split / f"{i:06d}.txt").write_text("\n".join(lines))


def write_yaml(out: Path):
    names = "\n".join(f'  {i}: "{c}"' for i, c in enumerate(CLASSES))  # "-" "/" "=" 는 따옴표 필수
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(CLASSES)}\nnames:\n{names}\n"
    )


def selftest():
    import tempfile
    random.seed(0)
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as td:
        pool = fake_pool(Path(td))
        for _ in range(200):
            img, boxes = render_sample(pool)
            W, H = img.size
            assert len(boxes) >= 1, "박스가 하나도 안 생김"
            for b in boxes:
                cid, cx, cy, bw, bh = to_yolo(b, W, H)
                assert 0 <= cid < len(CLASSES)
                assert 0.0 < cx < 1.0 and 0.0 < cy < 1.0, "중심이 이미지 밖"
                assert 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0, "박스 크기 이상"
                _, x1, y1, x2, y2 = b
                assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H, "박스가 이미지 밖"
    print("selftest OK: 라벨 지오메트리 정상")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="xainano 클래스별 폴더 경로")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=4000, help="train 장수 (val=10%)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.dataset:
        ap.error("--dataset 또는 --selftest 필요")

    pool = load_pool(args.dataset)
    out = Path(args.out)
    write_split(pool, out, args.n, "train")
    write_split(pool, out, max(1, args.n // 10), "val")
    write_yaml(out)
    print(f"완료: {out}/  (train={args.n}, val={args.n // 10})")
    print(f"학습:  yolo detect train model=yolov8n.pt data={out}/data.yaml imgsz=128 epochs=60")


if __name__ == "__main__":
    main()
