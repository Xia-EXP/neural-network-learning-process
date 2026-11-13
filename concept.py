import base64, colorsys, io  # Encode probe thumbnails and convert nominal probe colors for display.
import numpy as np  # Measure taught colors and recolor object pixels.
from PIL import Image, ImageDraw, ImageFilter  # Build masks, feathered edges, and fallback probe objects.
from scipy import ndimage  # Keep the largest connected colored region as the probed object.
from vision import features  # Query exactly the same feature pipeline used by training and inference.
from calibration import corrected  # Probe in the same corrected color space the model was trained on.

HUES = list(range(0, 360, 10))  # Sweep the complete hue circle in thirty-six equal steps.
TONES = [dict(key='vivid', name='鲜艳', s=.82, v=.86), dict(key='dark', name='深色', s=.80, v=.42), dict(key='light', name='浅色', s=.32, v=.96), dict(key='muted', name='灰淡', s=.20, v=.62)]  # Probe each hue at distinct saturation and brightness levels.
NEUTRALS = [dict(name='白色', s=0, v=.95), dict(name='灰色', s=0, v=.55), dict(name='黑色', s=0, v=.14)]  # Include colorless objects that were never part of a hue category.

def hsv_array(image):  # Read hue, saturation, and value as floating point arrays in the 0-1 range.
    return np.asarray(image.convert('HSV'), dtype=np.float32) / 255.0  # Use PIL's HSV conversion shared by the color-feature extractor.

def hue_weights(hsv):  # Emphasize saturated, reasonably bright pixels that carry object color.
    return hsv[..., 1] ** 2 * np.sqrt(hsv[..., 2]) * (hsv[..., 1] > .18) * (hsv[..., 2] > .1)  # Suppress gray background and deep shadows while keeping muted object colors.

def hue_histogram(images):  # Measure the share of colorful evidence falling into each ten-degree hue bin.
    histogram = np.zeros(36)  # Accumulate weighted evidence in ten-degree bins.
    for image in images:  # Combine every photograph taught for this class.
        hsv = hsv_array(image)  # Measure actual pixel colors.
        histogram += np.histogram(hsv[..., 0] * 360, bins=36, range=(0, 360), weights=hue_weights(hsv))[0]  # Add hue evidence weighted by colorfulness.
    return histogram / max(float(histogram.sum()), 1e-8)  # Normalize so classes with more photographs are comparable.

def class_hues(groups):  # Estimate each class's distinctive hue; skin, clothing, and background shared by all classes cancel out.
    histograms = [hue_histogram(images) for images in groups]  # Measure every class's hue distribution.
    result = []  # Collect one hue estimate per class.
    for index, histogram in enumerate(histograms):  # Compare each class with the others.
        others = np.max([h for j, h in enumerate(histograms) if j != index], axis=0)  # Find hues that also appear in other classes.
        distinct = histogram * np.maximum(np.log((histogram + .002) / (others + .002)), 0)  # Weight each hue by its share and by how specific it is to this class.
        smoothed = distinct + .5 * np.roll(distinct, 1) + .5 * np.roll(distinct, -1)  # Smooth neighboring bins with circular wrapping.
        if smoothed.max() < .02: result.append(None); continue  # Report colorless or indistinct classes instead of inventing a hue.
        peak = int(np.argmax(smoothed))  # Locate the strongest distinctive hue bin.
        bins = np.arange(peak - 2, peak + 3) % 36  # Refine the estimate using the neighboring bins.
        angles, weights = np.radians(bins * 10 + 5), distinct[bins]  # Weight neighboring bin centers by distinctive evidence.
        result.append(float(np.degrees(np.arctan2((np.sin(angles) * weights).sum(), (np.cos(angles) * weights).sum())) % 360))  # Use a circular mean in degrees.
    return result  # Return hues in class order.

def object_mask(image, hue):  # Locate the taught object's colored pixels in one photograph.
    if hue is None: return None  # Leave colorless classes to the fallback probe object.
    hsv = hsv_array(image)  # Measure the photograph's actual colors.
    distance = np.abs((hsv[..., 0] * 360 - hue + 180) % 360 - 180)  # Measure circular hue distance from the class color.
    near = (distance < 24) & (hsv[..., 1] > .2) & (hsv[..., 2] > .12)  # Select colorful pixels near the class hue.
    if not near.any(): return None  # Report photographs where the class color is absent.
    mask = near & (hsv[..., 1] > max(.2, .6 * float(np.percentile(hsv[..., 1][near], 90))))  # Separate the vivid object from paler skin of a similar hue.
    mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))) > 127  # Remove isolated speckles.
    labels, count = ndimage.label(mask)  # Separate disconnected colored regions.
    if count: mask = labels == (1 + int(np.argmax(ndimage.sum(mask, labels, range(1, count + 1)))))  # Keep only the largest region, which is the held object.
    filled = ndimage.binary_fill_holes(mask)  # Include printed logos and highlights inside the object outline.
    return Image.fromarray((filled * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))  # Return an 8-bit object mask for feathered recoloring.

def fallback_mask(size):  # Supply a centered disc when no colored object can be isolated.
    mask = Image.new('L', size, 0)  # Start with an empty mask.
    width, height = size  # Read the teaching-region dimensions.
    ImageDraw.Draw(mask).ellipse([width * .32, height * .26, width * .68, height * .74], fill=255)  # Draw a cap-sized disc in the center.
    return mask  # Return a mask with the same contract as a measured object mask.

ACHROMATIC = .12  # Below this median saturation a region carries no usable relative-saturation pattern, so scaling it cannot add color.

def recolor(image, mask, hue, saturation, value, measured=True):  # Repaint only masked pixels while keeping shape, shading, hands, and background.
    hsv = hsv_array(image)  # Measure the original pixels.
    inside = np.asarray(mask, dtype=np.float32) > 127  # Read the binary object region.
    if not inside.any(): mask, measured = fallback_mask(image.size), False; inside = np.asarray(mask, dtype=np.float32) > 127  # Never return an unchanged photo for an empty mask; use the disclosed artificial disc instead.
    median_s = float(np.median(hsv[..., 1][inside]))  # Measure typical saturation inside the repainted region.
    median_v = float(np.median(hsv[..., 2][inside]))  # Measure typical brightness inside the repainted region.
    painted = hsv.copy()  # Keep the original photograph unchanged.
    painted[..., 0] = hue / 360  # Replace the hue with the probe hue.
    if not saturation: painted[..., 1] = 0  # Neutral probes (white, gray, black) carry no hue at all.
    elif measured and median_s >= ACHROMATIC: painted[..., 1] = np.clip(hsv[..., 1] * saturation / median_s, 0, 1)  # Colored measured objects keep their relative saturation shading.
    else: painted[..., 1] = saturation  # Gray objects and fallback discs have zero saturation to scale, so assign the probe saturation explicitly.
    if measured and median_v >= .05: painted[..., 2] = np.clip(hsv[..., 2] * value / median_v, 0, 1)  # Scale a measured object's brightness to the probe tone while preserving highlights and shadows.
    else: painted[..., 2] = value  # A disclosed artificial disc (or a near-black region) has no usable shading, so paint the intended probe brightness flat.
    repainted = Image.fromarray(np.clip(painted * 255, 0, 255).astype(np.uint8), 'HSV').convert('RGB')  # Convert the repainted pixels back to RGB.
    feather = mask.filter(ImageFilter.GaussianBlur(1.2))  # Soften the boundary between repainted and original pixels.
    return Image.composite(repainted, image, feather)  # Blend the new object color into the unchanged scene.

def thumbnail(image, size=(96, 72), quality=72):  # Encode a compact probe picture for the interface.
    output = io.BytesIO()  # Prepare an in-memory JPEG.
    image.resize(size).save(output, 'JPEG', quality=quality)  # Keep thumbnails small enough for the one-time map download.
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode()  # Return a data URL usable by an image element.

def hex_color(hue, saturation, value):  # Express a nominal probe color for interface swatches.
    return '#' + ''.join(f'{round(c * 255):02x}' for c in colorsys.hsv_to_rgb(hue / 360, saturation, value))  # Convert HSV to a CSS hexadecimal color.

def taught_profile(images, hue):  # Summarize the colors students actually showed for one class.
    histogram, total, pixels = np.zeros(36), np.zeros(3), 0  # Accumulate hue evidence, color sums, and object pixel counts.
    for image in images:  # Inspect every taught photograph.
        hsv, rgb = hsv_array(image), np.asarray(image, dtype=np.float32)  # Read both color representations.
        located = object_mask(image, hue) if hue is not None else None  # Locate the object when the class has a color.
        mask = np.asarray(located, dtype=np.float32) > 127 if located is not None else np.zeros(hsv.shape[:2], bool)  # Skip photographs where the object is absent.
        histogram += np.histogram(hsv[..., 0][mask] * 360, bins=36, range=(0, 360), weights=hsv[..., 1][mask] ** 2)[0]  # Record which hues the object pixels contained.
        total += rgb[mask].sum(axis=0); pixels += int(mask.sum())  # Accumulate the object's average color.
    swatch = '#' + ''.join(f'{int(c):02x}' for c in (total / pixels if pixels else [160, 166, 162]))  # Report the measured mean object color or a neutral placeholder.
    peak = float(histogram.max()) or 1.0  # Normalize the hue histogram for drawing.
    return dict(hue=None if hue is None else round(hue, 1), swatch=swatch, hist=np.round(histogram / peak, 3).tolist(), pixels=pixels, images=len(images))  # Return a JSON-safe description of the taught color.

def answer(scores):  # Package one real probability vector as a winning label plus rounded scores.
    return dict(p=int(np.argmax(scores)), s=np.round(scores, 3).tolist())  # Keep the exact model output that the interface displays.

def build(model, samples, calibration, decode, progress, cancelled, fit=None):  # Probe the trained model with repainted copies of real training photographs.
    by_class = [[corrected(decode(s['image'])[1], calibration) for s in samples if s['label'] == label] for label in range(3)]  # Use the same corrected pixels the model learned from.
    hues = class_hues(by_class)  # Estimate each taught class color from what distinguishes its photographs.
    taught = [taught_profile(images, hue) for images, hue in zip(by_class, hues)]  # Describe what each class actually looked like.
    bases = []  # Collect one representative photograph and object mask per class.
    for label, (images, hue) in enumerate(zip(by_class, hues)):  # Choose the photograph whose object is most clearly visible, remembering its class.
        best = None  # Track the strongest candidate photograph.
        for image in images[:40]:  # Bound the search on large classes.
            mask = object_mask(image, hue)  # Locate the colored object.
            area = float(np.mean(np.asarray(mask) > 127)) if mask is not None else 0.0  # Measure the visible object fraction.
            if best is None or area > best[2]: best = (image, mask, area)  # Keep the clearest object.
        if best and best[2] >= .015: bases.append(dict(image=best[0], mask=best[1], measured=True, label=label, area=best[2]))  # Use a measured object mask when one is large enough.
        elif images: bases.append(dict(image=images[0], mask=fallback_mask(images[0].size), measured=False, label=label, area=0.0))  # Fall back to a disclosed artificial centered disc for colorless or unlocatable objects.
    original = [answer(model.predict_proba([features(b['image'])])[0]) for b in bases]  # Ask the model about each unmodified base photograph, a real original-sample measurement.
    total = len(bases) * (len(HUES) * len(TONES) + len(NEUTRALS))  # Count every real model query in advance for progress reporting.
    done, grid, thumbs = 0, [], []  # Track progress, results, and every displayed probe picture.
    def ask(hue, tone):  # Query the model once per repainted base photograph and keep each answer.
        nonlocal done  # Update shared progress from the nested helper.
        per_base, raw, pictures = [], [], []  # Keep every contributing answer, its full-precision scores, and its exact probe picture for auditing.
        for base in bases:  # Repaint each representative photograph.
            probe = recolor(base['image'], base['mask'], hue, tone['s'], tone['v'], base['measured'])  # Change only the object (or disclosed fallback disc) color.
            scores = model.predict_proba([features(probe)])[0]  # Query the trained model through the shared feature pipeline.
            raw.append(scores); per_base.append(answer(scores))  # Keep full precision for averaging and round only the serialized per-base answer.
            pictures.append(thumbnail(probe, (80, 60), 70))  # Keep the very picture behind this answer.
            done += 1  # Count this real query.
        if cancelled(): raise InterruptedError  # Stop promptly when the lesson changes.
        progress(done, total)  # Publish real progress.
        mean = np.mean(raw, axis=0) if raw else np.zeros(3)  # Average full-precision outputs so rounding cannot change the aggregate winner near ties.
        return dict(answer(mean), b=per_base), pictures  # Return the aggregate together with every contributing per-base answer.
    for tone in TONES:  # Probe every tone row.
        row, row_thumbs = [], []  # Collect one cell and its probe pictures per hue.
        for hue in HUES:  # Sweep the hue circle.
            cell, pictures = ask(hue, tone)  # Ask the model about this repainted color on every base photograph.
            cell['hex'] = hex_color(hue, tone['s'], tone['v'])  # Attach the nominal probe color.
            row.append(cell); row_thumbs.append(pictures)  # Keep answers and pictures aligned in hue order.
        grid.append(row); thumbs.append(row_thumbs)  # Store the completed tone row and its pictures.
    neutrals = []  # Collect answers for colorless probes.
    for neutral in NEUTRALS:  # Ask about white, gray, and black objects.
        cell, pictures = ask(0, neutral)  # Use zero saturation so hue is irrelevant.
        neutrals.append(dict(name=neutral['name'], hex=hex_color(0, 0, neutral['v']), thumbs=pictures, **cell))  # Keep every measured answer with its probe picture.
    base_info = [dict(label=b['label'], measured=b['measured'], area=round(b['area'], 4), thumb=thumbnail(b['image']), original=o) for b, o in zip(bases, original)]  # Describe each base photograph, its mask type, and the model's answer on the untouched photo.
    return dict(status='done', done=done, total=total, hues=HUES, tones=[dict(key=t['key'], name=t['name'], s=t['s'], v=t['v']) for t in TONES], grid=grid, thumbs=thumbs, neutrals=neutrals, taught=taught, bases=base_info, measured=[b['measured'] for b in bases], fit=fit)  # Return the complete measured color map with per-base evidence and real original-sample fit.
