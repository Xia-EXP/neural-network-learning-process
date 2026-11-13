import numpy as np  # Measure neutral-reference pixels and apply identical RGB correction to every model input.
from PIL import Image  # Return standard RGB images accepted by the existing feature extractor.

def validate_calibration(value):  # Validate saved reference measurements before using imported correction gains.
    if value is None: return None  # Keep older projects compatible without inventing a calibration.
    if not isinstance(value, dict) or value.get('source') not in ['camera', 'simulation']: raise ValueError('颜色校准来源无效')  # Preserve camera and simulated provenance.
    gains, means = value.get('gains'), value.get('means')  # Read only the supported numeric calibration fields.
    if not isinstance(gains, list) or len(gains) != 3 or any(type(v) not in [int, float] or not np.isfinite(v) or not .625 <= v <= 1.6 for v in gains): raise ValueError('颜色校准增益无效')  # Reject unsafe or nonfinite correction multipliers.
    if not isinstance(means, list) or len(means) != 3 or any(type(v) not in [int, float] or not np.isfinite(v) or not 60 <= v <= 240 for v in means): raise ValueError('白色参考测量无效')  # Bound stored neutral-reference exposure values.
    return dict(source=value['source'], gains=list(gains), means=list(means))  # Keep only validated and portable measurement data.

def measure_reference(image, source):  # Estimate neutral balance from a white or neutral gray reference covering the selected region.
    pixels = np.asarray(image, dtype=np.float32)  # Preserve the original camera channel measurements.
    height, width = pixels.shape[:2]  # Locate the central reference patch away from teaching-region edges.
    patch = pixels[height//5:height*4//5, width//5:width*4//5]  # Measure the central sixty percent of the selected region.
    means = patch.mean(axis=(0, 1))  # Average RGB measurements to reduce small sensor noise.
    if means.min() < 60 or means.max() > 240 or np.mean((patch < 8) | (patch > 248)) > .02: raise ValueError('参考物过暗或过曝，请调整光照后重试')  # Reject reference readings with inadequate exposure headroom.
    if float(patch.std(axis=(0, 1)).max()) > 22: raise ValueError('参考区域不均匀，请让白纸铺满教学区并避开阴影')  # Reject object edges, strong shadows, and mixed reference surfaces.
    gains = means.mean() / means  # Neutralize channel imbalance without claiming calibrated colorimetry or changing exposure.
    if gains.min() < .625 or gains.max() > 1.6: raise ValueError('偏色过大，请确认使用白纸或中性灰卡')  # Reject strongly colored objects and extreme correction demands.
    return validate_calibration(dict(source=source, means=means.tolist(), gains=gains.tolist()))  # Reuse the same validation for measured and imported references.

def corrected(image, calibration):  # Apply one frozen reference profile to training and prediction alike.
    if calibration is None: return image  # Preserve all original behavior when no reference has been measured.
    pixels = np.asarray(image, dtype=np.float32) * np.asarray(calibration['gains'], dtype=np.float32)  # Scale RGB channels using the actual neutral-reference measurement.
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))  # Clamp corrected pixels to the valid RGB range.
