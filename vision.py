import hashlib, threading  # Identify repeated images and serialize the lazy model initialization.
from collections import OrderedDict  # Bound the in-memory feature cache without retaining camera images.
from pathlib import Path  # Locate the bundled pretrained weights independently of the current directory.
import numpy as np  # Normalize image tensors and aggregate color statistics.
import onnx  # Select the pretrained network's pooled visual features instead of its original labels.
import onnxruntime as ort  # Run the compact frozen backbone on the CPU without a training framework.
from PIL import Image, ImageEnhance, ImageOps  # Apply label-preserving exposure and orientation variations.

MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'mobilenetv2-7.onnx'  # Keep the approximately 14MB pretrained model offline.
MODEL_HASH = 'c1c513582d56afceff8516c73804e484c81c6a830712ab6d682253f4a3cd042f'  # Verify the downloaded model against the inspected source artifact.
SESSION = None  # Load the backbone only when a lesson actually trains or predicts.
GUARD = threading.RLock()  # Protect model initialization and the shared bounded feature cache.
CACHE = OrderedDict()  # Cache only feature vectors keyed by image digests, not the source photographs.
MODEL_NAME = 'MobileNetV2 + 小型分类头'  # Describe frozen feature extraction and learned classroom labels accurately.

def session():  # Initialize the CPU inference engine once for all classroom requests.
    global SESSION  # Publish the initialized engine for reuse by training and inference.
    with GUARD:  # Prevent concurrent requests from loading duplicate networks.
        if SESSION is None:  # Avoid parsing and optimizing the network on every image.
            if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_HASH: raise ValueError('轻量视觉模型缺失或校验失败')  # Refuse damaged or unexpectedly replaced weights.
            graph = onnx.load(str(MODEL_PATH))  # Read the bundled ONNX graph and pretrained parameters.
            pooled = next(node.output[0] for node in reversed(graph.graph.node) if node.op_type == 'GlobalAveragePool')  # Locate the actual position-pooled visual embedding.
            graph.graph.output.clear()  # Remove ImageNet's unrelated thousand-class output.
            graph.graph.output.append(onnx.helper.make_tensor_value_info(pooled, onnx.TensorProto.FLOAT, [1, 1280, 1, 1]))  # Expose the pretrained 1280-dimensional representation.
            initialized = {item.name for item in graph.graph.initializer}  # Identify legacy weights incorrectly listed as runtime inputs.
            inputs = [item for item in graph.graph.input if item.name not in initialized]  # Keep only image data as the required model input.
            graph.graph.input.clear()  # Replace the legacy input declaration with a modern inference declaration.
            graph.graph.input.extend(inputs)  # Preserve the original image-input tensor definition.
            options = ort.SessionOptions()  # Configure inference for an ordinary office computer.
            options.intra_op_num_threads = 1  # Avoid CPU oversubscription during frequent small predictions.
            options.inter_op_num_threads = 1  # Keep graph execution deterministic and resource usage modest.
            options.log_severity_level = 3  # Suppress harmless graph-pruning messages in the launcher.
            SESSION = ort.InferenceSession(graph.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])  # Freeze the backbone and run it entirely locally.
        return SESSION  # Reuse the same inference engine for subsequent images.

def color_features(image):  # Summarize color independently of object position without fixed color-to-label rules.
    hsv = np.asarray(image.resize((96, 72)).convert('HSV'), dtype=np.float32) / 255.0  # Separate hue from brightness and saturation.
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]  # Read the color channels as floating point arrays.
    weights = saturation ** 3 * np.sqrt(value) * (value > .08) * (saturation > .12)  # Reduce gray backgrounds, dark shadows, and low-saturation skin contributions.
    histogram = np.histogram(hue, bins=24, range=(0, 1.00001), weights=weights)[0]  # Describe observed hues without assigning any semantic category names.
    histogram = histogram + .2 * np.roll(histogram, 1) + .2 * np.roll(histogram, -1)  # Smooth neighboring hues with red wrapping around the circular hue axis.
    histogram = np.sqrt(histogram / max(float(histogram.sum()), 1e-8))  # Make color evidence insensitive to the object's pixel area.
    return histogram.astype(np.float32)  # Return a compact position-independent color descriptor.

def features(image):  # Combine transferable visual features with exposure-tolerant color evidence.
    rgb = image.convert('RGB')  # Ensure camera and augmented inputs use identical channel ordering.
    key = hashlib.sha256(rgb.tobytes() + str(rgb.size).encode()).digest()  # Cache by actual image pixels and dimensions rather than a source label.
    with GUARD:  # Check cached embeddings without racing another request.
        if key in CACHE: CACHE.move_to_end(key); return CACHE[key]  # Reuse previously computed features while maintaining cache recency.
    pixels = np.asarray(rgb.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0  # Preserve the complete selected region rather than cropping away off-center objects.
    normalized = (pixels - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)  # Apply the backbone's ImageNet RGB normalization.
    tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])  # Convert the image to the model's NCHW tensor layout.
    engine = session()  # Lazily initialize the verified local backbone.
    embedding = engine.run(None, {engine.get_inputs()[0].name: tensor})[0].reshape(-1)  # Extract learned visual patterns instead of raw screen coordinates.
    embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-8)  # Normalize embedding scale across exposures and object sizes.
    result = np.concatenate([embedding * 3, color_features(rgb) * 2]).astype(np.float32)  # Balance general visual features with explicit color evidence for the existing color lesson.
    with GUARD:  # Publish the computed vector and bound cache memory.
        CACHE[key] = result  # Retain only the digest and feature vector.
        if len(CACHE) > 1024: CACHE.popitem(last=False)  # Evict the least recently used vector when the cache fills.
    return result  # Share exactly the same representation between classroom fitting and prediction.

def training_views(image):  # Produce modest augmentation without changing a student's class label.
    return [image, ImageEnhance.Brightness(image).enhance(.78), ImageEnhance.Brightness(ImageOps.mirror(image)).enhance(1.18)]  # Expose the classifier to lighting changes and horizontal placement variation.

