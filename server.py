from __future__ import annotations  # Allow forward-compatible type annotations.
import base64, binascii, hashlib, io, json, os, secrets, socket, uuid, threading, time, webbrowser  # Provide image identity, LAN request tokens, and robust input handling.
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer  # Serve the offline application and its API.
from pathlib import Path  # Resolve assets relative to this file, independent of the launch directory.
from urllib.parse import parse_qs, urlsplit  # Read token query parameters without treating the URL path as trusted input.
import numpy as np  # Represent image features and training targets.
from PIL import Image  # Decode and resize camera snapshots consistently.
from sklearn.neural_network import MLPClassifier  # Use a maintained neural-network implementation for real training.
from threadpoolctl import threadpool_limits  # Keep the small model from consuming every CPU thread.
from vision import features, training_views, MODEL_NAME  # Reuse frozen visual features while training a fresh classroom neural-network head.
from calibration import corrected, measure_reference, validate_calibration  # Separate neutral-reference correction from student-provided semantic labels.
import concept  # Probe the trained model with repainted training photographs to reveal its learned color definitions.
from dataset_quality import inspect_samples  # Identify exact repeated frames and contradictory labels before real training.
from dataset_quality import fingerprint  # 以一致的真实像素标识关联样本、固定检验照片与历史轮次。
from learning_history import new_run, comparison, check_result, validate_learning  # 统一管理可核实的训练摘要与独立检验对比。

ROOT = Path(__file__).resolve().parent  # Locate the application folder.
DATA = Path(os.environ.get('LEARNING_DESK_DATA', str(ROOT / 'data' / 'session.json')))  # Isolate automated test storage when explicitly configured.
LOCK = threading.RLock()  # Serialize access to the dataset and published model.
LABELS = ['红色', '黄色', '蓝色']  # Fix the output-class order across training and inference.
STATE = dict(samples=[], names=LABELS.copy(), curve=[], trace=[], training=False, ready=False, error='', generation=0, elapsed=0, model=None, phase='', extracted=0, extraction_total=0, paused=False, step_budget=0, pace_ms=150)  # Preserve genuine per-round measurements and explicit classroom pacing controls.
STATE.update(calibration=None, evaluations=[], concept=None)  # Retain the current reference profile, fresh-frame results, and the measured color map of the current model.
STATE.update(runs=[], tests=[], run_id=None, history_revision=0, pause_ack=False, extraction_index=None, extraction_view=0, benchmark_done=0, benchmark_total=0)  # 分开保存持久训练摘要、独立检验照片以及安全暂停位置。
LIVE = dict(image=None, source=None, updated=0)  # Hold only the latest shared teaching-region frame in volatile memory for the Android touch panel.
LAN_MODE = '--lan' in os.sys.argv  # LAN exposure is opt-in and must be visible in the launch command.
LAN_TOKEN = os.environ.get('LEARNING_DESK_LAN_TOKEN', '').strip() or (secrets.token_urlsafe(24) if LAN_MODE else '')  # Require a per-launch token when the service leaves localhost.

def lan_request_token(path, headers):  # Accept the token from the header during API calls or from the initial page URL.
    query = parse_qs(urlsplit(path).query)
    return headers.get('X-Learning-Token', '') or (query.get('token') or [''])[0]

def decode_image(value, keep=False):  # Validate a browser snapshot before feature extraction or storage; keep preserves the live-view size.
    if not isinstance(value, str) or len(value) > 160000: raise ValueError('图片过大或格式不正确')  # Bound image input size.
    try:
        raw = base64.b64decode(value.split(',', 1)[-1], validate=True)  # Decode the data URL strictly.
        with Image.open(io.BytesIO(raw)) as image:  # Close the decoder after checking image dimensions.
            if image.width > 640 or image.height > 480: raise ValueError('图片尺寸超过限制')  # Reject oversized decoded images.
            unchanged = image.format == 'JPEG' and image.mode == 'RGB' and (keep or image.size == (192, 144))  # 已经符合采集规范的照片无需再次有损编码。
            rgb = image.convert('RGB') if keep else image.convert('RGB').resize((192, 144))  # Normalize stored snapshots to a compact color image, keeping larger live frames sharp for the big screen.
    except (binascii.Error, OSError, ValueError) as error:
        if str(error) == '图片尺寸超过限制': raise
        raise ValueError('图片格式不正确') from error
    if unchanged: return 'data:image/jpeg;base64,' + base64.b64encode(raw).decode(), rgb  # 保持像素身份稳定，确保重复检测和训练泄漏检查可复现。
    output = io.BytesIO()  # Prepare an in-memory JPEG file.
    rgb.save(output, 'JPEG', quality=80)  # Store the actual captured image rather than fabricated thumbnails.
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode(), rgb  # Return a normalized snapshot and decoded pixels.

def snapshot(include_samples=False):  # Produce a JSON-safe view without exposing the Python model object.
    with LOCK:  # Keep counts and training progress from different updates consistent.
        result = {key: STATE[key] for key in ['names', 'curve', 'trace', 'training', 'ready', 'error', 'elapsed', 'generation', 'phase', 'extracted', 'extraction_total', 'paused', 'pace_ms', 'calibration', 'evaluations']}  # Return measured learning, calibration, and fresh-frame results without executable model objects.
        result['model_name'] = MODEL_NAME  # Let the interface identify the pretrained backbone and trained classifier honestly.
        result['counts'] = [sum(s['label'] == i for s in STATE['samples']) for i in range(3)]  # Count the real captured images per class.
        result['quality'] = inspect_samples(STATE['samples'])  # Expose cached pixel-based data checks so every exhibit screen shows the same evidence.
        result['sources'] = sorted(set(s['source'] for s in STATE['samples']))  # Make simulated and camera datasets distinguishable.
        result['concept'] = None if STATE['concept'] is None else {key: STATE['concept'].get(key) for key in ['status', 'done', 'total', 'generation', 'error']}  # Report color-map progress without polling its thumbnails.
        result.update({key: STATE[key] for key in ['pause_ack', 'extraction_index', 'extraction_view', 'benchmark_done', 'benchmark_total', 'run_id']})  # 将暂停确认、当前处理照片和固定检验进度同步给主屏。
        measured = comparison(STATE['runs'], STATE['tests'], STATE['samples']); measured.pop('rows')  # 轮询只传对比计数，补样中的训练泄漏即时排除。
        result['learning_summary'] = dict(revision=STATE['history_revision'], runs=len(STATE['runs']), tests=len(STATE['tests']), comparison=measured)  # 避免每次轮询都传输历史图片。
        if include_samples: result.update(version=1, samples=list(STATE['samples']), learning=dict(runs=list(STATE['runs']), tests=list(STATE['tests'])))  # 课堂导出同时保留训练摘要和独立检验集。
        return result  # Return an independent response object.

def persist():  # Save the current dataset atomically so an interrupted write preserves the previous file.
    DATA.parent.mkdir(parents=True, exist_ok=True)  # Create the local data folder on first use.
    temporary = DATA.with_name(DATA.stem+'-'+uuid.uuid4().hex+'.tmp')  # Use a unique local filename without an unbounded permission-error retry loop.
    created = False  # Clean up only a file this save successfully created.
    try:  # Close the newly created file before replacing the dataset on Windows.
        with temporary.open('x', encoding='utf-8') as output:  # Fail promptly on permission errors and never overwrite another writer's temporary file.
            created = True;json.dump(dict(version=1, names=STATE['names'], samples=STATE['samples'], calibration=STATE['calibration'], learning=dict(runs=STATE['runs'], tests=STATE['tests'])), output, ensure_ascii=False);output.flush();os.fsync(output.fileno())  # 原子保存照片、标定、训练摘要与固定检验集。
        os.replace(temporary, DATA)  # Publish only the fully written session snapshot.
    finally:  # Remove only this save attempt's uncommitted file after an error.
        if created and temporary.exists(): temporary.unlink()  # Leave the last successful saved dataset intact.

def validate_project(project):  # Check imported and restored projects before replacing the active session.
    if not isinstance(project, dict) or project.get('version') != 1: raise ValueError('不是本工具的项目文件')  # Reject unknown project formats.
    names = project.get('names')  # Read user-defined category names.
    if not isinstance(names, list) or len(names) != 3 or any(not isinstance(n, str) or not n.strip() or len(n) > 12 for n in names): raise ValueError('需要三个不超过12字的类别名称')  # Bound category labels.
    if len(set(n.strip() for n in names)) != 3: raise ValueError('三个类别名称不能相同')  # Avoid ambiguous class names.
    samples = project.get('samples')  # Read the exported snapshot records.
    if not isinstance(samples, list) or len(samples) > 360: raise ValueError('每个项目最多360张样本')  # Bound memory and disk use.
    clean = []  # Accumulate fully validated records before mutating live state.
    for sample in samples:  # Validate every snapshot independently.
        if not isinstance(sample, dict) or type(sample.get('label')) is not int or sample['label'] not in range(3): raise ValueError('样本类别无效')  # Enforce the fixed class index range.
        if sample.get('source') not in ['camera', 'simulation']: raise ValueError('样本来源无效')  # Preserve explicit source provenance.
        image, _ = decode_image(sample.get('image'))  # Decode and normalize imported images.
        clean.append(dict(label=sample['label'], source=sample['source'], image=image))  # Retain only the supported record fields.
    if any(sum(s['label'] == i for s in clean) > 120 for i in range(3)): raise ValueError('每类最多120张样本')  # Enforce the same cap used during capture.
    if len({sample['source'] for sample in clean}) > 1: raise ValueError('项目不能混合摄像头与模拟样本')  # Apply the same source separation to imported projects and startup recovery.
    return [n.strip() for n in names], clean  # Return a safe dataset ready to publish.

def invalidate():  # Cancel an old training job and prevent inference against obsolete data.
    STATE.update(generation=STATE['generation'] + 1, model=None, ready=False, training=False, curve=[], trace=[], evaluations=[], concept=None, error='', elapsed=0, phase='', extracted=0, extraction_total=0, paused=False, step_budget=0, pause_ack=False, extraction_index=None, extraction_view=0, benchmark_done=0, benchmark_total=0, run_id=None)  # 撤销旧模型和活动记录，同时保留独立的已完成历史供补样对比。

def commit_project(reset_model=True, **changes):  # Publish a project edit only when its durable save succeeds; callers hold LOCK throughout.
    previous = dict(STATE)  # Preserve the prior samples, model and training generation for rollback without copying the model.
    try:  # Keep memory and the on-disk classroom project consistent across a failed write.
        STATE.update(changes)  # Replace edited collections instead of mutating the previous snapshot in place.
        persist()  # Atomically save the edited project before acknowledging it to any screen.
        if reset_model: invalidate()  # Publish cancellation only after saving succeeds so lock-free concept checks cannot observe a rolled-back generation.
    except OSError:  # Restore the exact previous lesson if the filesystem rejects the save.
        STATE.clear(); STATE.update(previous); raise  # Retain a usable prior model and dataset while returning the original storage error.

def wait_turn(generation, allow_step=True):  # 在真实计算边界暂停，特征提取和固定检验不消费学习单步预算。
    while True:  # Recheck both reset and teacher commands during a pause.
        with LOCK:  # Read one consistent control state for the current lesson.
            if generation != STATE['generation']: return False  # Cancel a waiting round when the lesson is reset.
            if not STATE['paused']: STATE['pause_ack'] = False; return True  # 恢复后继续实际计算并清除已停下标记。
            if allow_step and STATE['phase'] == 'fit' and STATE['step_budget'] > 0: STATE['step_budget'] -= 1; STATE['pause_ack'] = False; return True  # 每次单步只放行一轮真实参数更新。
            STATE['pause_ack'] = True  # 只有计算已经到达安全边界，才向界面确认已暂停。
        time.sleep(.02)  # Yield the CPU while the teacher discusses the current record.

def round_record(model, x, y, probes, probe_indices, inputs, previous, before, epoch):  # Inspect the trained network without altering its library-managed optimizer.
    hidden_indices = np.linspace(0, 31, 8, dtype=int)  # Show a fixed subset of the actual thirty-two hidden units.
    activations = np.maximum(0, probes @ model.coefs_[0] + model.intercepts_[0])  # Read the actual ReLU hidden activations for the three representative samples.
    after = model.predict_proba(probes)  # Measure the network's output after this actual optimization round.
    correct = int(np.sum(model.predict(x[::3]) == y[::3]))  # Evaluate only original captured frames, excluding their augmentation copies.
    probabilities = model.predict_proba(x[::3]); guesses = model.classes_[np.argmax(probabilities, axis=1)]; original_labels = y[::3]  # 在同一批原图上读取真实概率，答对统计与分类损失共用这次预测。
    by_class = [dict(label=label, correct=int(np.sum((original_labels == label) & (guesses == label))), total=int(np.sum(original_labels == label)), loss=float(-np.log(np.clip(probabilities[original_labels == label, label], 1e-12, 1)).mean())) for label in range(3)]  # 各类曲线采用原图真实交叉熵，答对之后仍能看见模型得分继续学习。
    parameters = model.coefs_ + model.intercepts_  # Include both connection weights and biases in the update measurement.
    delta = None if previous is None else float(np.sqrt(sum(np.sum((new-old)**2) for new, old in zip(parameters, previous))))  # Measure real parameter movement since the previous recorded round.
    return dict(epoch=epoch+1, loss=float(model.loss_), correct=correct, total=len(y[::3]), by_class=by_class, wrong_indices=np.flatnonzero(guesses != original_labels).tolist(), delta=delta, before=None if before is None else np.round(before, 5).tolist(), after=np.round(after, 5).tolist(), sample_indices=probe_indices, inputs=np.round(probes[:, inputs], 4).tolist(), hidden=np.round(activations[:, hidden_indices], 4).tolist(), weights_in=np.round(model.coefs_[0][np.ix_(inputs, hidden_indices)], 4).tolist(), weights_out=np.round(model.coefs_[1][hidden_indices], 4).tolist())  # 每轮保存真实样本关联、更新前后得分、分类统计与抽样网络测量。

def train(generation, samples, calibration=None):  # Freeze the same reference profile throughout one background training job.
    started = time.perf_counter()  # Measure real computation time rather than an animated countdown.
    try:  # Publish failures through the API instead of silently losing the worker.
        vectors, labels = [], []  # Accumulate pretrained features without changing or replacing stored camera photographs.
        with LOCK:  # Publish the genuine preprocessing workload before extracting features.
            if generation != STATE['generation']: return  # Honor a reset that occurred before the worker began.
            STATE.update(phase='extract', extraction_total=len(samples) * 3)  # Count the original view and two label-preserving augmentations per sample.
        for sample_index, sample in enumerate(samples):  # 让界面可对应当前正在处理的真实照片。
            for view_index, view in enumerate(training_views(corrected(decode_image(sample['image'])[1], calibration))):  # 为原图与两个增强视图提供实际处理序号。
                if not wait_turn(generation, allow_step=False): return  # 每个视觉特征计算开始前响应暂停和重置。
                with LOCK:  # Check cancellation before each neural-network feature extraction.
                    if generation != STATE['generation']: return  # Stop immediately when another lesson supersedes this dataset.
                    STATE.update(extraction_index=sample_index, extraction_view=view_index)  # 指出当前照片及其原图、变暗或镜像增强视图。
                vectors.append(features(view)); labels.append(sample['label'])  # Extract visual patterns using the frozen pretrained backbone.
                with LOCK:  # Publish only this job's measured progress and elapsed time.
                    if generation != STATE['generation']: return  # Avoid publishing stale extraction counters after reset.
                    STATE.update(extracted=len(vectors), elapsed=round(time.perf_counter() - started, 3))  # Report completed real image processing work.
        x, y = np.array(vectors), np.array(labels)  # Assemble the augmented feature matrix and unchanged class labels.
        if not wait_turn(generation, allow_step=False): return  # 最后一个特征算完后仍保持请求中的暂停状态。
        with LOCK:  # Distinguish frozen visual extraction from the genuinely learned classifier head.
            if generation != STATE['generation']: return  # Recheck cancellation before classifier optimization.
            STATE['phase'] = 'fit'  # Mark the start of actual class-learning passes.
        model = MLPClassifier(hidden_layer_sizes=(32,), solver='adam', learning_rate_init=0.003, alpha=0.01, batch_size=min(32, len(y)), random_state=42)  # Train a small CPU model from fresh weights each time.
        probe_indices = [next(i for i, sample in enumerate(samples) if sample['label'] == label) for label in range(3)]  # Select one actual classroom photograph per category for consistent round-to-round comparison.
        probes = x[np.array(probe_indices) * 3]  # Locate each representative original view in the augmented feature matrix.
        input_indices = np.argsort(np.var(x, axis=0))[-6:]  # Display six real feature channels that vary across the taught examples.
        with threadpool_limits(limits=1):  # Keep training efficient on ordinary office computers.
            for epoch in range(80):  # Produce eighty actual optimization passes.
                if not wait_turn(generation): return  # Honor pause, single-step, resume, and reset before the next real pass.
                with LOCK:  # Check whether reset or dataset edits superseded this job.
                    if generation != STATE['generation']: return  # Discard a cancelled job without publishing stale results.
                previous = [value.copy() for value in model.coefs_ + model.intercepts_] if epoch else None  # Snapshot existing parameters only after the library has actually initialized the network.
                before = model.predict_proba(probes) if epoch else None  # Avoid inventing an unobserved pre-initialization prediction for round one.
                model.partial_fit(x, y, classes=np.arange(3))  # Continue optimization without restarting model weights.
                record = round_record(model, x, y, probes, probe_indices, input_indices, previous, before, epoch)  # Measure activations, output changes, fit counts, and weight updates after this pass.
                with LOCK:  # Publish each measured loss safely to the polling interface.
                    if generation != STATE['generation']: return  # Recheck cancellation after the optimization step.
                    STATE['curve'].append(float(model.loss_))  # Record the library's real training loss.
                    STATE['trace'].append(record)  # Keep an immutable history for honest post-training replay and scrubbing.
                    STATE['elapsed'] = round(time.perf_counter() - started, 3)  # Record elapsed wall-clock training time.
                    pace = STATE['pace_ms'] / 1000  # Read the explicitly selected presentation interval between actual updates.
                deadline = time.perf_counter() + pace  # Space real optimization passes without substituting an animated fake training loop.
                while epoch < 79 and time.perf_counter() < deadline:  # Leave the final completed pass immediately available for inference.
                    with LOCK:  # Allow resets and changed classroom controls to interrupt presentation pacing.
                        if generation != STATE['generation']: return  # Cancel without publishing an old model after reset.
                        if STATE['paused'] or STATE['pace_ms'] == 0: break  # Respond promptly to pause or fast-mode commands.
                    time.sleep(.01)  # Keep deliberate classroom pacing lightweight and interruptible.
        with LOCK:  # 冻结本轮完成后的固定检验工作量。
            if generation != STATE['generation']: return  # 不使用已经过期的训练结果。
            trained_digests = {fingerprint(sample['image']) for sample in samples}  # 训练过的照片不得作为独立检验。
            tests = [test for test in STATE['tests'] if test['digest'] not in trained_digests and test['source'] == samples[0]['source']]  # 只自动重测来源一致且未进入本轮训练的照片。
            STATE.update(phase='check', benchmark_total=len(tests), benchmark_done=0)  # 实际优化完成后单独显示固定检验阶段。
        checks = []  # 累积同一批固定检验照片的真实结果。
        for test in tests:  # 使用新模型重新回答之前保存的独立照片。
            if not wait_turn(generation, allow_step=False): return  # 固定检验也遵守真实暂停边界。
            checks.append(check_result(test, model.predict_proba([features(corrected(decode_image(test['image'])[1], calibration))])[0]))  # 与实时推理共用相同的图像预处理和特征函数。
            with LOCK:  # 逐张发布实际完成的检验数量。
                if generation != STATE['generation']: return  # 拒绝旧任务的检验进度。
                STATE['benchmark_done'] = len(checks)  # 不把固定检验计入训练轮次或人工新照片检验次数。
        with LOCK:  # 原子保存完成摘要后才发布可用模型。
            if generation != STATE['generation']: return  # Never publish a model for a superseded lesson.
            run = new_run(samples, STATE['names'], calibration, STATE['curve'], STATE['trace'], checks)  # 形成独立于活动模型的历史记录。
            commit_project(reset_model=False, runs=(STATE['runs'] + [run])[-6:], run_id=run['id'], history_revision=STATE['history_revision'] + 1)  # 最近六轮摘要随课堂文件持久保存，写入失败时不发布新历史。
            STATE.update(model=model, ready=True, training=False, phase='done', paused=False, pause_ack=False, concept=dict(status='running', done=0, total=0, generation=generation))  # 真实训练与固定检验均完成后开放实时识别。
        guesses = model.predict(x[::3])  # Re-answer every original (non-augmented) training photograph with the finished model.
        fit = [dict(label=label, correct=int(np.sum((y[::3] == label) & (guesses == label))), total=int(np.sum(y[::3] == label))) for label in range(3)]  # Count real per-class original-sample fit, distinct from synthetic probes and fresh examinations.
        probe_concept(generation, model, samples, calibration, fit)  # Measure what each learned category means across the color space.
    except Exception as error:  # Keep worker errors visible and allow another training attempt.
        with LOCK:  # Avoid overwriting a newer session's state with an older failure.
            if generation == STATE['generation']: STATE.update(training=False, ready=False, error=str(error))  # Report the failing job's error.

def probe_concept(generation, model, samples, calibration, fit=None):  # Ask the finished model about repainted photographs while live inference stays available.
    def progress(done, total):  # Publish real query counts for the interface progress ring.
        with LOCK:  # Update only this lesson's measurement record.
            if generation == STATE['generation'] and STATE['concept']: STATE['concept'].update(done=done, total=total)  # Ignore progress from a superseded model.
    try:  # Keep a probing failure from affecting the trained model or inference.
        result = concept.build(model, samples, calibration, decode_image, progress, lambda: generation != STATE['generation'], fit)  # Run every probe through the shared feature pipeline and attach the real original-sample fit.
        with LOCK:  # Publish the completed map atomically.
            if generation == STATE['generation']: STATE['concept'] = dict(result, generation=generation)  # Bind the map to the model that produced it.
    except InterruptedError: return  # Abandon measurements for a model that was replaced.
    except Exception as error:  # Report unexpected probing failures without hiding them.
        with LOCK:  # Avoid overwriting a newer lesson.
            if generation == STATE['generation']: STATE['concept'] = dict(status='error', error=str(error), done=0, total=0, generation=generation)  # Surface the failure in the interface.

class LocalServer(ThreadingHTTPServer):  # Prevent Windows from silently sharing an already occupied application port.
    allow_reuse_address = False  # Force another instance to select the next free port instead of sharing requests.
    def server_bind(self):  # Apply exclusive socket ownership before binding the HTTP service.
        if os.name == 'nt': self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)  # Keep test and classroom datasets isolated on Windows.
        super().server_bind()  # Bind only after the socket policy is configured.

class Handler(SimpleHTTPRequestHandler):  # Combine static assets and a small local JSON API.
    def __init__(self, *args, **kwargs): super().__init__(*args, directory=str(ROOT / 'web'), **kwargs)  # Expose only public web assets.
    def log_message(self, *args): pass  # Keep repeated inference requests from flooding the launch window.
    def request_path(self): return urlsplit(self.path).path  # Route by the path component so a token query does not bypass an endpoint.
    def authorized(self): return not LAN_MODE or secrets.compare_digest(lan_request_token(self.path, self.headers), LAN_TOKEN)  # Protect every LAN API read and write with the same per-launch secret.
    def reply(self, payload, status=200):  # Send UTF-8 JSON with explicit caching and length headers.
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')  # Preserve Chinese labels in API responses.
        try:  # A visitor may navigate away while a camera inference response is being sent.
            self.send_response(status)  # Report the appropriate HTTP status.
            self.send_header('Content-Type', 'application/json; charset=utf-8')  # Tell the browser how to decode the payload.
            self.send_header('Content-Length', str(len(body)))  # Delimit the response body correctly.
            self.send_header('Cache-Control', 'no-store')  # Prevent stale model and dataset state.
            self.end_headers()  # Finish the HTTP headers before writing the body.
            self.wfile.write(body)  # Deliver the serialized response.
        except ConnectionError: self.close_connection = True  # End a disconnected response without attempting a second HTTP error response.
    def do_GET(self):  # Return live state, exported projects, or application assets.
        path = self.request_path()
        if path.startswith('/api/') and not self.authorized(): return self.reply(dict(error='局域网访问需要有效令牌'), 403)  # Do not expose classroom images or controls to unauthenticated LAN clients.
        if path == '/api/live':  # Share a transient station-camera crop without requiring Android camera permissions over HTTP.
            with LOCK: return self.reply(dict(**LIVE, age=time.time()-LIVE['updated']))  # Include freshness so remote controls cannot label a disconnected stale image.
        if path == '/api/connection': return self.reply(dict(lan=LAN_MODE, port=self.server.server_port, addresses=lan_addresses(), token=LAN_TOKEN if LAN_MODE else ''))  # Advertise the token needed by the optional touch panel.
        if path == '/api/status': return self.reply(snapshot())  # Poll training progress without transferring images.
        if path == '/api/learning':  # 按需读取已完成轮次和逐张固定检验证据。
            with LOCK: return self.reply(dict(runs=list(STATE['runs']), tests=list(STATE['tests']), comparison=comparison(STATE['runs'], STATE['tests'], STATE['samples']), revision=STATE['history_revision'], run_id=STATE['run_id'], generation=STATE['generation'], in_training=[test['id'] for test in STATE['tests'] if any(fingerprint(sample['image']) == test['digest'] for sample in STATE['samples'])]))  # 历史详情同时指出已补入当前训练样本的检验照片。
        if path == '/api/concept':  # Return the complete measured color map for the current model.
            with LOCK: return self.reply(dict(concept=STATE['concept'], names=STATE['names'], generation=STATE['generation']))  # Include names so labels match the probed classes.
        if path == '/api/project': return self.reply(snapshot(True))  # Restore the full classroom session.
        if path.startswith('/api/'): return self.reply(dict(error='接口不存在'), 404)  # Reject misspelled API paths clearly.
        return super().do_GET()  # Serve local HTML, JavaScript, styles, and icons.
    def do_POST(self):  # Validate and dispatch browser commands.
        try:  # Convert malformed requests into readable errors.
            path = self.request_path()
            if path.startswith('/api/') and not self.authorized(): return self.reply(dict(error='局域网访问需要有效令牌'), 403)  # Apply the same protection to sample, training, and reset mutations.
            origin = self.headers.get('Origin')  # Inspect the browser origin before allowing mutations.
            if origin and origin != 'http://' + self.headers.get('Host', ''): return self.reply(dict(error='请求来源不匹配'), 403)  # Accept only the application's own origin.
            length = int(self.headers.get('Content-Length', '0'))  # Read the declared JSON body size.
            if length < 1 or length > 25000000: raise ValueError('请求内容为空或过大')  # Bound import and snapshot requests.
            body = json.loads(self.rfile.read(length))  # Parse structured input rather than manipulating JSON strings.
            if not isinstance(body, dict): raise ValueError('请求必须为对象')  # Ensure all route handlers receive a mapping.
            with LOCK:  # Apply each command as one consistent state transition.
                if path == '/api/live':  # Update the volatile frame relayed from the station browser.
                    if body.get('source') not in ['camera', 'simulation']: raise ValueError('画面来源无效')  # Keep remote labels associated with the real active source.
                    image, _ = decode_image(body.get('image'), keep=True)  # Bound and validate the shared teaching-region image.
                    LIVE.update(image=image, source=body['source'], updated=time.time())  # Replace the previous frame without persisting camera video.
                    return self.reply(dict(ok=True))  # Acknowledge the small periodic relay request.
                if path == '/api/calibrate':  # Measure a neutral reference independently of semantic sample labels.
                    if STATE['training']: raise ValueError('请先结束训练再校准颜色')  # Avoid changing preprocessing underneath an optimizer.
                    source = body.get('source')  # Associate the profile with its camera or simulated source.
                    if source not in ['camera', 'simulation'] or any(s['source'] != source for s in STATE['samples']): raise ValueError('参考物与现有样本来源不一致')  # Prevent cross-source color correction.
                    _, image = decode_image(body.get('image'))  # Read the actual reference image supplied by the touch action.
                    profile = measure_reference(image, source)  # Reject poor exposure, uneven patches, and extreme channel imbalance.
                    commit_project(calibration=profile)  # Save changed preprocessing transactionally and require fresh training.
                    return self.reply(snapshot())  # Display measured channel values and applied gains.
                if path == '/api/calibration-clear':  # Restore raw-color processing explicitly.
                    if STATE['training']: raise ValueError('请先结束训练再修改校准')  # Preserve training-input consistency.
                    commit_project(calibration=None)  # Remove the profile only after a successful save and invalidate its old model.
                    return self.reply(snapshot())  # Publish the uncalibrated state.
                if path == '/api/evaluate':  # Examine a fresh frozen frame against the student's independently supplied expected label.
                    if not STATE['ready']: raise ValueError('请先完成训练')  # Never invent a test outcome before a usable model exists.
                    if body.get('generation') != STATE['generation']: raise ValueError('模型已变化，请重新拍摄检验画面')  # Reject a challenge captured under an obsolete model.
                    label, source = body.get('label'), body.get('source')  # Read the truth chosen before revealing the answer.
                    if type(label) is not int or label not in range(3) or source not in ['camera', 'simulation'] or any(s['source'] != source for s in STATE['samples']): raise ValueError('检验类别或来源无效')  # Match examination provenance to the lesson.
                    image, pixels = decode_image(body.get('image'))  # Freeze and retain the actual tested view for the result thumbnail.
                    digest = fingerprint(image)  # 使用与训练样本一致的像素身份检查重复和训练泄漏。
                    if any(record['digest'] == digest for record in STATE['evaluations']): raise ValueError('这张画面已经检验过，请重新摆放或换样品')  # Prevent repeated clicks on one frozen view from inflating the result count.
                    if any(fingerprint(sample['image']) == digest for sample in STATE['samples']): raise ValueError('这张画面已用于训练，请重新摆放样品再检验')  # 明确排除已用于训练的完全相同画面。
                    if len(STATE['evaluations']) >= 60: raise ValueError('本轮已完成60次检验，请导出结果或重新训练')  # Bound the in-memory examination history.
                    scores = STATE['model'].predict_proba([features(corrected(pixels, STATE['calibration']))])[0].tolist()  # Use precisely the trained model and frozen preprocessing.
                    record = dict(label=label, predicted=int(np.argmax(scores)), scores=scores, image=image, source=source, digest=digest)  # Keep a genuine deduplicated observation separate from training samples.
                    test = next((item for item in STATE['tests'] if item['digest'] == digest), None)  # 同一张固定检验照片保留稳定身份。
                    if test and (test['label'] != label or test['source'] != source): raise ValueError('这张固定检验照片已有不同标记，请换一张新照片')  # 避免固定检验答案在两轮之间被悄悄改写。
                    if test is None and len(STATE['tests']) >= 60: raise ValueError('固定检验集已满60张，请先保存课堂或重新教学')  # 为持久检验照片设置明确容量。
                    test = test or dict(id=uuid.uuid4().hex, digest=digest, label=label, source=source, image=image)  # 新拍照检验同时成为后续重训可复测的独立照片。
                    tests = STATE['tests'] if any(item['id'] == test['id'] for item in STATE['tests']) else STATE['tests'] + [test]  # 保持检验集一张画面只计一次。
                    runs = [dict(run, checks=[item for item in run['checks'] if item['id'] != test['id']] + [check_result(test, scores)]) if run['id'] == STATE['run_id'] else run for run in STATE['runs']]  # 将这次真实答案关联到产生它的完成轮次。
                    commit_project(reset_model=False, tests=tests, runs=runs, evaluations=STATE['evaluations'] + [record], history_revision=STATE['history_revision'] + 1)  # 保存检验集和结果后才确认拍照检验成功。
                    return self.reply(snapshot())  # Present cumulative results and the revealed model answer.
                if path == '/api/sample':  # Capture a labeled frame from the selected source.
                    if 'generation' in body and body['generation'] != STATE['generation']: raise ValueError('样本或标定已变化，请重新拍摄')  # Prevent a frozen capture from receiving labels under a different concurrent lesson state.
                    label, source = body.get('label'), body.get('source')  # Read class and capture-source metadata.
                    if type(label) is not int or label not in range(3) or source not in ['camera', 'simulation']: raise ValueError('类别或图像来源无效')  # Reject invalid capture metadata.
                    if STATE['training']: raise ValueError('训练中不能追加样本')  # Keep the training dataset immutable during a job.
                    if any(s['source'] != source for s in STATE['samples']): raise ValueError('摄像头与模拟样本不能混用，请先清空或导出项目')  # Prevent simulated evidence from masquerading as camera performance.
                    if STATE['calibration'] and STATE['calibration']['source'] != source: raise ValueError('当前颜色校准来自另一画面来源，请先清除校准')  # Prevent the first photo from silently inheriting a calibration measured on another source.
                    if sum(s['label'] == label for s in STATE['samples']) >= 120: raise ValueError('该类已满120张')  # Bound repetitive capture.
                    image, _ = decode_image(body.get('image'))  # Validate the actual teaching-region image.
                    sample = dict(label=label, source=source, image=image)  # Store label and provenance beside the image.
                    commit_project(samples=STATE['samples'] + [sample])  # Save the added photograph and invalidate the old model as one recoverable transaction.
                    return self.reply(dict(sample=sample, state=snapshot()))  # Update the browser without reloading all thumbnails.
                if path == '/api/sample-edit':  # Let visitors correct the actual human-provided labels rather than forcing model answers.
                    if STATE['training']: raise ValueError('请先完成训练，再修改标定')  # Keep the active optimization dataset immutable.
                    if type(body.get('generation')) is not int or body['generation'] != STATE['generation']: raise ValueError('样本已变化，请重新打开这张照片')  # Reject stale indices after another screen adds or edits photographs.
                    index, action, label = body.get('index'), body.get('action'), body.get('label')  # Read one explicit sample mutation and its semantic target.
                    if type(index) is not int or not 0 <= index < len(STATE['samples']): raise ValueError('照片不存在')  # Reject negative and out-of-range indexing.
                    if action not in ['label', 'delete']: raise ValueError('样本操作无效')  # Restrict the endpoint to two reviewable dataset actions.
                    updated = list(STATE['samples'])  # Stage the reviewed edit without modifying the previous lesson's sample list.
                    if action == 'label':  # Reassign only the selected captured photograph.
                        if type(label) is not int or label not in range(3): raise ValueError('类别无效')  # Preserve the three declared visitor classes.
                        if STATE['samples'][index]['label'] == label: return self.reply(snapshot(True))  # Preserve a valid model and scores when the visitor confirms the existing label.
                        if sum(s['label'] == label for s in STATE['samples']) >= 120: raise ValueError('目标类别已满120张')  # Enforce the same class capacity as normal sample collection.
                        updated[index] = dict(STATE['samples'][index], label=label)  # Change the semantic label without decoding or modifying the photograph or provenance.
                    else: updated.pop(index)  # Delete only the exact sample the visitor reviewed.
                    commit_project(samples=updated)  # Publish the reviewed edit and require retraining only after the save succeeds.
                    return self.reply(snapshot(True))  # Refresh thumbnails, counts, quality checks and the genuine model state together.
                if path == '/api/train':  # Start a fresh training run from the current samples.
                    if STATE['training']: raise ValueError('当前正在训练')  # Prevent duplicate worker jobs.
                    if inspect_samples(STATE['samples'])['conflicts']: raise ValueError('同一张照片被标成了不同类别，请先检查并修正标定')  # Block mathematically contradictory supervision rather than pretending additional optimizer passes can fix it.
                    if any(count < 3 for count in inspect_samples(STATE['samples'])['unique_per_class']): raise ValueError('每类至少采集3张不同画面，完全重复的照片不增加有效张数')  # 原始重复张数不能代替各类别的不同训练画面。
                    pace_ms = body.get('pace_ms', 0)  # Keep API automation fast while the classroom UI requests its teaching interval explicitly.
                    if type(pace_ms) is not int or pace_ms not in [0, 150, 350]: raise ValueError('训练节奏无效')  # Restrict deliberate pacing to the supported classroom choices.
                    invalidate()  # Clear the previous model and create a unique job generation.
                    STATE.update(training=True, pace_ms=pace_ms)  # Publish the chosen real training pace before starting the worker.
                    threading.Thread(target=train, args=(STATE['generation'], list(STATE['samples']), STATE['calibration']), daemon=True).start()  # Freeze calibration alongside the samples without blocking HTTP.
                    return self.reply(snapshot())  # Immediately return the new job state.
                if path == '/api/train-control':  # Apply teacher controls to actual optimization rather than just the drawing.
                    if not STATE['training']: raise ValueError('当前没有正在进行的训练')  # Keep completed-history replay separate from live training controls.
                    action = body.get('action')  # Read the explicit pause, resume, step, or speed command.
                    if action == 'pause': STATE.update(paused=True, pause_ack=False, step_budget=0)  # 先请求暂停，到达真实计算边界后才确认已停下。
                    elif action == 'resume': STATE.update(paused=False, pause_ack=False, step_budget=0)  # 从实际停下的位置继续。
                    elif action == 'step' and STATE['paused'] and STATE['pause_ack'] and STATE['phase'] == 'fit': STATE['step_budget'] = 1; STATE['pause_ack'] = False  # 已确认暂停时恰好放行一轮优化。
                    elif action == 'pace' and type(body.get('pace_ms')) is int and body['pace_ms'] in [0, 150, 350]: STATE['pace_ms'] = body['pace_ms']  # Update the interval without changing the learning algorithm.
                    else: raise ValueError('训练控制命令无效')  # Reject unsupported or inappropriate state transitions.
                    return self.reply(snapshot())  # Return current live control state to the teacher.
                if path == '/api/predict':  # Run genuine inference with the current trained model.
                    if not STATE['ready']: raise ValueError('请先完成训练')  # Never fabricate predictions before training.
                    _, image = decode_image(body.get('image'))  # Decode the same input region used during capture.
                    vector = features(corrected(image, STATE['calibration']))  # Apply exactly the same neutral-reference correction as training.
                    scores = STATE['model'].predict_proba([vector])[0].tolist()  # Obtain scores for the student's three named categories.
                    return self.reply(dict(scores=scores, generation=STATE['generation']))  # Preserve genuine closed-set output for the neural-network learning demonstration.
                if path == '/api/reset':  # 原子清空当前课堂配置中的照片、标定和学习历史。
                    if 'generation' in body and (type(body['generation']) is not int or body['generation'] != STATE['generation']): raise ValueError('样本已变化，请重新选择删除范围')  # 拒绝跨屏样本变化后的旧清理请求。
                    if 'revision' in body and (type(body['revision']) is not int or body['revision'] != STATE['history_revision']): raise ValueError('检验记录已变化，请重新选择删除范围')  # 独立检验新增时不能沿用旧确认。
                    if body.get('delete_all') and STATE['training']: raise ValueError('请先完成训练，再删除样本')  # 删除入口在实际训练中保持锁定，原有重新教学仍可取消整轮。
                    commit_project(samples=[], calibration=None, runs=[], tests=[], history_revision=STATE['history_revision'] + 1)  # 直接替换活动配置，不另存旧照片文件夹。
                    return self.reply(snapshot(True))  # Return the empty dataset to the interface.
                if path == '/api/delete':  # 按类删除保留历史，明确全部删除则彻底清空当前课堂。
                    if STATE['training']: raise ValueError('请先完成训练，再删除样本')  # 保证正在学习的照片集合在整轮训练中保持不变。
                    label = body.get('label')  # 接受既有的数值类别，以及明确选择的全部样本范围。
                    if label != 'all' and (type(label) is not int or label not in range(3)): raise ValueError('删除范围无效')  # 缺少范围、布尔值和越界标签都不能触发清理。
                    if label == 'all' and 'generation' not in body: raise ValueError('请重新选择要删除的样本')  # 全部删除必须基于浏览器已确认的样本版本。
                    if 'generation' in body and (type(body['generation']) is not int or body['generation'] != STATE['generation']): raise ValueError('样本已变化，请重新选择删除范围')  # 另一屏添加或修改照片后，拒绝旧确认请求。
                    if label == 'all':  # 全部范围与页面重置入口保持同样的永久清理含义。
                        if 'revision' in body and (type(body['revision']) is not int or body['revision'] != STATE['history_revision']): raise ValueError('检验记录已变化，请重新选择删除范围')  # 对携带历史版本的请求验证独立照片是否变化。
                        commit_project(samples=[], calibration=None, runs=[], tests=[], history_revision=STATE['history_revision'] + 1)  # 原子移除训练与检验照片、标定和摘要，失败时整体回滚。
                        return self.reply(snapshot(True))  # 发布空课堂并清除活动模型和训练轨迹。
                    remaining = [sample for sample in STATE['samples'] if label != 'all' and sample['label'] != label]  # 一次构造删除后的完整集合，全部清理也只进行一次原子写入。
                    if len(remaining) == len(STATE['samples']): return self.reply(snapshot(True))  # 空类别无需写盘，也不撤销当前有效模型。
                    commit_project(samples=remaining)  # 保存成功后再发布剩余照片和待重训状态，保存失败则完整回滚。
                    return self.reply(snapshot(True))  # 一并更新照片带、质量计数和当前模型状态。
                if path == '/api/names':  # Rename the three classes without changing their numeric labels.
                    names, _ = validate_project(dict(version=1, names=body.get('names'), samples=[]))  # Reuse project-label validation.
                    commit_project(reset_model=False, names=names)  # Save the new display names without discarding a valid classifier, rolling back on failure.
                    return self.reply(snapshot())  # Refresh labels while preserving the trained classifier.
                if path == '/api/import':  # Restore an exported project for another lesson.
                    names, samples = validate_project(body)  # Validate the complete replacement before touching live data.
                    runs, tests = validate_learning(body.get('learning'), decode_image)  # 兼容旧文件并验证新增的训练历史与固定检验照片。
                    profile = validate_calibration(body.get('calibration'))  # Validate optional reference gains before replacing any project state.
                    if profile and any(s['source'] != profile['source'] for s in samples): raise ValueError('校准与样本来源不一致')  # Reject imported preprocessing for another source.
                    commit_project(names=names, samples=samples, calibration=profile, runs=runs, tests=tests, history_revision=STATE['history_revision'] + 1)  # 导入整份课堂资料后重新训练活动模型，同时保留只读历史。
                    return self.reply(snapshot(True))  # Display the imported images and labels.
                return self.reply(dict(error='接口不存在'), 404)  # Reject unsupported commands.
        except (ValueError, TypeError, KeyError, OSError) as error: self.reply(dict(error=str(error)), 400)  # Surface input and storage errors to the operator.
        except Exception as error: self.reply(dict(error='处理失败：' + str(error)), 500)  # Keep unexpected failures visible without terminating the server.

def lan_addresses():  # Discover adapter addresses without contacting an external service.
    try: return sorted({entry[4][0] for entry in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET) if not entry[4][0].startswith('127.')})  # Exclude loopback addresses from Android connection choices.
    except OSError: return []  # Keep local use available even if adapter discovery fails.

def main():  # Restore local samples and launch an explicitly selected local or classroom-LAN service.
    if DATA.exists():  # Preserve earlier lessons when restarting the app.
        try:  # Fully validate the saved lesson before publishing any restored state.
            project = json.loads(DATA.read_text(encoding='utf-8')); names, samples = validate_project(project); profile = validate_calibration(project.get('calibration'))  # Stage validated names, raw photographs and optional reference gains.
            runs, tests = validate_learning(project.get('learning'), decode_image)  # 服务重启后恢复训练摘要和独立检验集。
            if profile and any(sample['source'] != profile['source'] for sample in samples): raise ValueError('校准与样本来源不一致')  # Reject mismatched saved preprocessing just as the import API does.
            STATE.update(names=names, samples=samples, calibration=profile, runs=runs, tests=tests)  # 完整验证通过才载入课堂，模型仍需本次启动后实际训练。
        except Exception as error: STATE['error'] = '历史样本未载入：' + str(error)  # Keep a damaged session file for investigation.
    service = None  # Track whether an available local port has been found.
    for port in range(8765, 8796):  # Use another port when an existing application occupies the first one.
        try: service = LocalServer(('0.0.0.0' if LAN_MODE else '127.0.0.1', port), Handler); break  # Enable the Android panel only in explicitly requested LAN mode.
        except OSError: continue  # Try the next local port.
    if service is None: raise RuntimeError('8765—8795端口均不可用')  # Explain the rare case where no local port is available.
    url = f'http://127.0.0.1:{service.server_port}' + (f'?token={LAN_TOKEN}' if LAN_MODE else '')  # Pass the per-launch token to the local browser when LAN mode is enabled.
    print(f'Learning Desk: {url}', flush=True)  # Make the chosen URL visible in the launch window.
    if '--no-browser' not in os.sys.argv: webbrowser.open(url)  # Open a browser during normal double-click launch.
    try: service.serve_forever()  # Keep the app available until the launch window is closed.
    except KeyboardInterrupt: pass  # Support a clean Ctrl+C stop during maintenance.
    finally: service.server_close()  # Release the listening socket on exit.

if __name__ == '__main__': main()  # Start the application only when invoked as a program.

