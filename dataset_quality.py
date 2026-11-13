import base64, hashlib, io  # Compare decoded captured pixels without relying on filenames or JPEG metadata.
from functools import lru_cache  # Reuse image fingerprints during frequent state polling.
from PIL import Image  # Decode the same real sample photographs used by training.

@lru_cache(maxsize=512)  # Bound retained image fingerprints and avoid decoding the complete dataset every poll.
def fingerprint(value):  # Identify an exact observed image independently of its assigned semantic label.
    with Image.open(io.BytesIO(base64.b64decode(value.split(',', 1)[-1]))) as image:  # Read the already validated stored camera or simulation snapshot.
        return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()  # Compare actual pixel content rather than confidence or assumed color names.

def inspect_samples(samples):  # Report exact label conflicts and repeated frames without claiming independent visual diversity.
    groups, unique = {}, [set(), set(), set()]  # Keep frame identities shared across classes and counted within each class.
    for index, sample in enumerate(samples):  # Preserve source indices so the visitor can inspect and repair the conflicting photographs.
        digest = fingerprint(sample['image']); groups.setdefault(digest, []).append((index, sample['label'])); unique[sample['label']].add(digest)  # Track exactly equal frames while leaving all original samples intact.
    conflicts = [dict(indices=[index for index, label in group], labels=sorted({label for index, label in group})) for group in groups.values() if len({label for index, label in group}) > 1]  # Flag only proven identical-image contradictions rather than guessing that similar objects are mislabeled.
    repeated_indices = sorted(index for group in groups.values() if len(group) > 1 for index, label in group)  # 返回完全相同照片的全部位置，便于从采集区直接检查。
    return dict(total=len(samples), unique=len(groups), unique_per_class=[len(group) for group in unique], repeats=len(samples)-len(groups), repeated_indices=repeated_indices, conflicts=conflicts)  # 同时提供原始张数、像素去重数量、重复照片位置和标签冲突。
