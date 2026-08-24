"""
FLIP 답안 크롭 -> 문자열 인식 + 채점.
학습된 YOLO-det 모델로 "답 칸 하나 크롭"을 읽는다. (전체 페이지 아님)

  python infer.py --model best.pt --image crop.jpg
  python infer.py --model best.pt --image crop.jpg --answer -3/4
  python infer.py --selftest        # 모델 없이 조립 로직만 검증
"""
import argparse

import numpy as np
from PIL import Image, ImageOps

from synth_data import CLASSES  # 클래스 순서 공유 (id -> 문자)

DISPLAY = {"pi": "π"}  # 클래스명 -> 표시/채점용 기호

# ── 로컬 테스트: 아래 3개만 바꾸고  `python infer.py`  실행 (CLI 인자로도 덮어씀) ──
MODEL = "/Users/akran/Downloads/best_256.pt"   # Colab 셀8로 받은 모델 경로
IMAGE = "test_images/KakaoTalk_Photo_2026-08-24-23-20-52.jpeg"   # test_images/ 에 사진 넣고 여기 파일명만 바꾸면 됨
ANSWER = "-367"                            # 채점하려면 "-367" 처럼, 인식만 볼거면 None


def disp(cls):
    return DISPLAY.get(cls, cls)


def assemble(dets):
    """검출 [(x_center, class_id, conf), ...] -> (문자열, 답신뢰도, [(문자,conf)]).

    x좌표 순 = 읽는 순서. 답 신뢰도는 제일 약한 글자 conf(weakest link).
    ponytail: min 휴리스틱. 자릿수 하나 틀리면 답 전체가 틀리니 최저값이 타당.
    """
    dets = sorted(dets, key=lambda d: d[0])
    per = [(disp(CLASSES[cid]), conf) for _, cid, conf in dets]
    text = "".join(c for c, _ in per)
    ans_conf = min((c for _, c in per), default=0.0)
    return text, ans_conf, per


def crop_to_ink(img):
    """테스트 편의용 자동 크롭: 채도 높은 잉크(색펜) 영역의 bbox로 자름.
    ponytail: 색펜+깨끗한 배경 가정(회색 그림자는 채도 낮아 무시됨). 흑펜이면
    잘 안 됨 -> 실제 제품은 답칸 템플릿 좌표로 크롭하므로 이건 테스트용."""
    a = np.asarray(img).astype(int)
    sat = a.max(2) - a.min(2)           # 채도: 색 있으면 큼, 회색(그림자)이면 작음
    ys, xs = np.where(sat > 35)
    if len(xs) < 50:                    # 잉크 못 찾으면 원본 그대로
        return img
    x1, x2 = int(np.percentile(xs, 2)), int(np.percentile(xs, 98))
    y1, y2 = int(np.percentile(ys, 2)), int(np.percentile(ys, 98))
    p = 30
    return img.crop((max(0, x1 - p), max(0, y1 - p),
                     min(img.width, x2 + p), min(img.height, y2 + p)))


def read_answer(model, image_path, conf=0.25, imgsz=128, autocrop=True):
    # imgsz는 학습과 동일해야 함 (기본 128). r 도 반환해 박스 저장에 씀.
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")  # 폰 사진 회전 보정
    if autocrop:
        img = crop_to_ink(img)
    # agnostic_nms=False(클래스별 NMS): 겹친 '다른 글자'(예: - 와 3)를 중복으로 안 지움.
    # 같은 글자 중복만 정리. 붙여쓴 답을 살리는 핵심 설정.
    r = model.predict(img, imgsz=imgsz, conf=conf, agnostic_nms=False, verbose=False)[0]
    dets = [(float(b.xywh[0][0]), int(b.cls), float(b.conf)) for b in r.boxes]
    text, ans_conf, per = assemble(dets)
    return text, ans_conf, per, r


def selftest():
    i2 = CLASSES.index("2"); islash = CLASSES.index("/"); i5 = CLASSES.index("5")
    ipi = CLASSES.index("pi")
    # 일부러 x 순서 뒤섞어 넣음 -> 정렬해서 "2/5" 나와야
    text, ac, per = assemble([(50, i5, 0.9), (10, i2, 0.8), (30, islash, 0.6)])
    assert text == "2/5", text
    assert abs(ac - 0.6) < 1e-9, ac          # 최저 conf
    assert per[0][0] == "2" and per[-1][0] == "5"
    assert assemble([(10, ipi, 0.7)])[0] == "π"   # pi -> π 매핑
    assert assemble([])[0] == "" and assemble([])[1] == 0.0
    print("selftest OK: 조립/정렬/신뢰도/π매핑 정상")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--answer", default=ANSWER, help="정답(주면 채점)")
    ap.add_argument("--conf", type=float, default=0.25, help="검출 최소 conf")
    ap.add_argument("--imgsz", type=int, default=256, help="추론 해상도 (학습과 동일! best_256->256)")
    ap.add_argument("--review", type=float, default=0.5, help="이 미만이면 사람 확인")
    ap.add_argument("--no-save", dest="save", action="store_false", help="박스 이미지 저장 안 함")
    ap.add_argument("--no-autocrop", dest="autocrop", action="store_false", help="자동 크롭 끔")
    ap.add_argument("--selftest", action="store_true")
    ap.set_defaults(save=True, autocrop=True)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.model or not args.image:
        ap.error("MODEL/IMAGE 를 파일 상단에 넣거나 --model --image 로 지정 (또는 --selftest)")

    import os
    from ultralytics import YOLO  # 지연 import: 학습 전에도 selftest 돌게
    model = YOLO(args.model)
    text, ans_conf, per, r = read_answer(model, args.image, args.conf, args.imgsz, args.autocrop)

    print(f"인식: {text!r}   신뢰도 {ans_conf:.0%}")
    print("  글자별:", ", ".join(f"{c}={p:.0%}" for c, p in per) or "(검출 없음)")
    if args.save:
        d = os.path.dirname(args.image) or "."   # 원본 옆(test_images/)에 저장
        out = os.path.join(d, "pred_" + os.path.basename(args.image))
        r.save(filename=out)
        print("  박스 이미지:", out)

    if ans_conf < args.review or not per:
        print("  -> ⚠ 선생님 확인 필요 (신뢰도 낮음)")
    elif args.answer is not None:
        ok = text == args.answer
        print(f"  -> 채점: {'O 정답' if ok else 'X 오답'}  (정답 {args.answer!r})")


if __name__ == "__main__":
    main()
