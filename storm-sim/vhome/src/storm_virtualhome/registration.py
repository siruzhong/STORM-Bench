"""Align nearby static features before comparing an inspected object location."""
import numpy as np


def align_region(before, after, region, excluded_before, excluded_after):
    import cv2

    cv2.setRNGSeed(42)
    yy, xx = np.where(region)
    if not len(xx):
        raise ValueError('The initial target region is empty')
    area = np.zeros(region.shape, dtype=np.uint8)
    area[max(0, yy.min() - 140):min(area.shape[0], yy.max() + 141),
         max(0, xx.min() - 140):min(area.shape[1], xx.max() + 141)] = 255
    source_mask, destination_mask = area.copy(), area.copy()
    kernel = np.ones((9, 9), np.uint8)
    source_mask[cv2.dilate((excluded_before | region).astype(np.uint8), kernel) > 0] = 0
    destination_mask[cv2.dilate(excluded_after.astype(np.uint8), kernel) > 0] = 0
    detector = cv2.ORB_create(nfeatures=4000, fastThreshold=8)
    a, da = detector.detectAndCompute(cv2.cvtColor(before, cv2.COLOR_RGB2GRAY), source_mask)
    b, db = detector.detectAndCompute(cv2.cvtColor(after, cv2.COLOR_RGB2GRAY), destination_mask)
    if da is None or db is None:
        raise ValueError('Insufficient static features for viewpoint alignment')
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < 12:
        raise ValueError('Too few static feature matches')
    source = np.float32([a[m.queryIdx].pt for m in good])
    destination = np.float32([b[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(source, destination, method=cv2.RANSAC,
                                               ransacReprojThreshold=2.0, maxIters=3000)
    if matrix is None or int(inliers.sum()) < 12:
        raise ValueError('Viewpoint alignment is not supported by enough inliers')
    scale = float(np.linalg.norm(matrix[:, 0]))
    if not 0.85 <= scale <= 1.15 or np.linalg.norm(matrix[:, 2]) > 100:
        raise ValueError('The inspection viewpoint is too different from the initial view')
    projected = source @ matrix[:, :2].T + matrix[:, 2]
    residual = float(np.median(np.linalg.norm(projected - destination, axis=1)[inliers.ravel() > 0]))
    warped = cv2.warpAffine(region.astype(np.uint8), matrix, (region.shape[1], region.shape[0]),
                            flags=cv2.INTER_NEAREST) > 0
    if warped.sum() < 0.7 * region.sum() or residual > 1.5:
        raise ValueError('Target region alignment failed')
    return warped, {'matrix': matrix.tolist(), 'inliers': int(inliers.sum()),
                    'median_residual_pixels': residual}
