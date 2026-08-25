"""스캔 세션: "찍고 -> 삐 -> 넘기고" 흐름의 비동기 채점.

캡처 체크(동기, 빠름)와 채점(비동기, 느림)을 분리한다:
- 삐 소리의 의미는 "잘 찍혔으니 넘겨" 뿐이다. 채점 완료가 아니다.
- 캡처를 통과한 페이지는 워커 풀에 들어가 병렬 채점되고,
  학생이 다음 장을 넘기는 동안 앞 장들이 뒤에서 처리된다.
- 체감 대기 = 마지막 장 +(페이지 1장 처리 시간), 장수와 무관.
"""
import concurrent.futures
import time
from pathlib import Path

import cv2

from flip.results import HOLD, O, X, format_page

# ── 캡처 체크 임계값 (튜닝 포인트) ──────────────────────────────────────
BLUR_MIN = 60.0        # 라플라시안 분산 하한. 미만이면 흔들림/초점 불량
PAGE_AREA_MIN = 0.30   # 페이지(밝은 영역)가 프레임에서 차지해야 하는 최소 비율
WORKERS = 4            # 채점 워커 수 (페이지 병렬)


def capture_check(image_path):
    """촬영 직후 빠른 판정: (통과 여부, 사유). 0.5초 이내 목표.

    통과 = 삐 (넘겨도 됨). 실패 = 재촬영 요구.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False, "이미지를 읽을 수 없음"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < BLUR_MIN:
        return False, f"흔들림/초점 불량 (blur {blur:.0f} < {BLUR_MIN:.0f})"

    # 페이지(밝은 픽셀)가 프레임에 충분히 들어왔는지
    _, th = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    page_ratio = (th > 0).mean()
    if page_ratio < PAGE_AREA_MIN:
        return False, f"페이지가 프레임에 부족 ({page_ratio:.0%} < {PAGE_AREA_MIN:.0%})"

    return True, f"OK (blur {blur:.0f}, page {page_ratio:.0%})"


class ScanSession:
    """페이지를 순서대로 투입받아 백그라운드에서 병렬 채점한다.

    grade_fn: (image_path) -> PageResult  (flip.grade.grade_page를 db 바인딩해 넘긴다)
    on_result: 페이지 결과가 나올 때마다 호출 (완료 순서, 투입 순서 아님)
    """

    def __init__(self, grade_fn, on_result=None):
        self.grade_fn = grade_fn
        self.on_result = on_result or (lambda pr: print(format_page(pr), flush=True))
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS)
        self.futures = []
        self.rejected = []  # (path, 사유)

    def submit(self, image_path):
        """캡처 체크 -> 통과 시 삐 + 채점 큐 투입. 반환: (통과, 사유)."""
        ok, reason = capture_check(image_path)
        if not ok:
            self.rejected.append((str(image_path), reason))
            print(f"[재촬영] {Path(image_path).name}: {reason}", flush=True)
            return False, reason
        print(f"[삐] {Path(image_path).name} — 넘기세요 ({reason})", flush=True)
        fut = self.pool.submit(self._grade_safe, image_path)
        fut.add_done_callback(lambda f: self.on_result(f.result()))
        self.futures.append(fut)
        return True, reason

    def _grade_safe(self, image_path):
        """한 페이지의 실패가 세션을 죽이지 않게 감싼다."""
        from flip.results import PageResult
        try:
            return self.grade_fn(image_path)
        except Exception as e:
            return PageResult(image=str(image_path), hold_reason=f"처리 오류: {e}")

    def finish(self):
        """모든 페이지 완료 대기 후 세션 요약 반환."""
        results = [f.result() for f in concurrent.futures.as_completed(self.futures)]
        self.pool.shutdown(wait=True)
        total = {O: 0, X: 0, HOLD: 0}
        for pr in results:
            for k, v in pr.counts().items():
                total[k] += v
        return results, total


def simulate(folder, grade_fn, interval=2.0):
    """폴더의 사진을 interval 간격으로 투입해 실시간 스캔을 재현 (--simulate).

    학생이 장을 넘기는 동안(interval) 앞 페이지들이 백그라운드에서 채점되는
    흐름을 데모로 보여준다.
    """
    images = sorted(p for p in Path(folder).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"이미지 없음: {folder}")

    session = ScanSession(grade_fn)
    t0 = time.time()
    for i, p in enumerate(images):
        session.submit(p)
        if i < len(images) - 1:
            time.sleep(interval)  # 장 넘기는 시간
    t_last = time.time()
    results, total = session.finish()
    t_end = time.time()

    print(f"\n== 세션 요약: {len(images)}장 투입, 거부 {len(session.rejected)}장")
    print(f"== O {total[O]} / X {total[X]} / 보류 {total[HOLD]}")
    print(f"== 총 {t_end - t0:.1f}s, 마지막 장 이후 대기 {t_end - t_last:.1f}s")
    return results


# ── selftest ─────────────────────────────────────────────────────────────

def _selftest():
    import tempfile

    import numpy as np

    from flip.results import PageResult, QuestionResult

    with tempfile.TemporaryDirectory() as tmp:
        # 선명한 페이지(텍스트 노이즈 포함) vs 민무늬(블러 판정) 이미지
        rng = np.random.default_rng(0)
        sharp = np.full((400, 300, 3), 255, np.uint8)
        for y in range(40, 360, 24):  # 텍스트 줄 흉내 (라플라시안 분산 확보)
            noise = (rng.random((8, 220)) > 0.5).astype(np.uint8) * 255
            sharp[y:y + 8, 40:260] = 255 - noise[..., None]
        flat = np.full((400, 300, 3), 255, np.uint8)
        cv2.imwrite(f"{tmp}/a_good.jpg", sharp)
        cv2.imwrite(f"{tmp}/b_flat.jpg", flat)

        ok, _ = capture_check(f"{tmp}/a_good.jpg")
        assert ok, "선명한 페이지는 캡처 체크 통과해야"
        ok2, reason2 = capture_check(f"{tmp}/b_flat.jpg")
        assert not ok2 and "흔들림" in reason2, reason2

        # 세션: 채점은 더미 grade_fn으로. 병렬 완료·집계·오류 격리 확인.
        collected = []

        def fake_grade(p):
            if "b_flat" in str(p):
                raise RuntimeError("의도된 실패")
            return PageResult(image=str(p), page_no="12",
                              results=[QuestionResult("1", "O")])

        session = ScanSession(fake_grade, on_result=collected.append)
        s1, _ = session.submit(f"{tmp}/a_good.jpg")
        assert s1
        # 캡처 체크에 걸리는 이미지는 큐에 안 들어간다
        s2, _ = session.submit(f"{tmp}/b_flat.jpg")
        assert not s2 and len(session.rejected) == 1
        results, total = session.finish()
        assert len(results) == 1 and total["O"] == 1
        assert len(collected) == 1

        # 오류 격리: grade_fn이 던져도 세션은 보류 결과로 흡수
        session2 = ScanSession(fake_grade, on_result=lambda pr: None)
        fut = session2.pool.submit(session2._grade_safe, f"{tmp}/b_flat.jpg")
        pr = fut.result()
        assert "처리 오류" in pr.hold_reason
        session2.pool.shutdown(wait=False)

    print("session selftest OK")


if __name__ == "__main__":
    _selftest()
