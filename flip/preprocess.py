"""페이지 사진 전처리: 원근 보정 + 명암 정규화.

폰으로 비스듬히 찍은 문제집 페이지를 "정면에서 본 종이"로 편다.
페이지 검출에 실패하면 원본 그대로 진행한다 (보정은 best-effort).
"""
import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없음: {path}")
    return img


def find_page_quad(img):
    """이미지에서 페이지(밝은 큰 사각형)의 네 귀퉁이를 찾는다. 실패 시 None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 배경(책상)보다 종이가 밝다는 가정. Otsu로 종이/배경 분리.
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < img.shape[0] * img.shape[1] * 0.3:
        return None  # 페이지가 프레임의 30% 미만이면 신뢰 안 함
    peri = cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, 0.02 * peri, True)
    if len(approx) != 4:
        return None
    return approx.reshape(4, 2).astype(np.float32)


def order_quad(quad):
    """네 점을 [좌상, 우상, 우하, 좌하] 순으로 정렬."""
    s = quad.sum(axis=1)
    d = np.diff(quad, axis=1).reshape(-1)
    return np.array([
        quad[np.argmin(s)],   # 좌상: x+y 최소
        quad[np.argmin(d)],   # 우상: y-x 최소
        quad[np.argmax(s)],   # 우하: x+y 최대
        quad[np.argmax(d)],   # 좌하: y-x 최대
    ], dtype=np.float32)


def warp_page(img, quad):
    """4점 원근 변환으로 페이지를 정면 뷰로 편다."""
    tl, tr, br, bl = order_quad(quad)
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), dst)
    return cv2.warpPerspective(img, m, (w, h))


def normalize_contrast(img):
    """조명 편차 완화: 그레이스케일 + CLAHE."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess(path):
    """이미지 경로 -> (보정된 컬러, 보정된 그레이). 파이프라인 1번."""
    img = load_image(path)
    quad = find_page_quad(img)
    if quad is not None:
        img = warp_page(img, quad)
    return img, normalize_contrast(img)
