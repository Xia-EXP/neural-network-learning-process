import hashlib, json, math, re, time, uuid  # 提供数据版本、历史结构检查、测量时间和独立记录标识。
from dataset_quality import fingerprint, inspect_samples  # 复用真实照片像素标识和采集质量统计。

def dataset_version(samples, calibration):  # 为每轮实际训练数据和预处理参数生成稳定版本。
    content = dict(samples=[dict(digest=fingerprint(sample['image']), label=sample['label'], source=sample['source']) for sample in samples], calibration=calibration)  # 标签、来源、照片或标定变化都属于新的训练输入。
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()  # 使用结构化序列化避免拼接字段产生歧义。

def new_run(samples, names, calibration, curve, trace, checks):  # 保存真实完成的一轮训练及固定检验结果。
    quality = inspect_samples(samples)  # 记录这轮使用的原始张数和去重数量。
    return dict(id=uuid.uuid4().hex, created=time.time(), dataset=dataset_version(samples, calibration), names=list(names), source=samples[0]['source'], counts=[sum(sample['label'] == label for sample in samples) for label in range(3)], unique_counts=quality['unique_per_class'], training_digests=sorted({fingerprint(sample['image']) for sample in samples}), curve=list(curve), fit=list(trace[-1]['by_class']), correct=trace[-1]['correct'], total=trace[-1]['total'], checks=list(checks))  # 只保存可检查的结果，不序列化可执行模型。

def comparison(runs, tests, samples=()):  # 只比较两轮共同测过且仍未进入当前训练样本的检验照片。
    if len(runs) < 2: return dict(status='waiting', common=0, excluded=0, before=None, after=None, rows=[], coverage=0)  # 两轮完成之前保留明确的等待状态。
    older, newer = runs[-2:]; previous = {item['id']: item for item in older['checks']}; current = {item['id']: item for item in newer['checks']}  # 按固定照片身份对应两轮输出。
    trained = set(older['training_digests']) | set(newer['training_digests']) | {fingerprint(sample['image']) for sample in samples}; rows = []; excluded = 0  # 补入当前样本的检验照片立即退出共同独立对比。
    for test in tests:  # 使用当前课堂实际保存的固定检验集。
        if test['digest'] in trained or test['source'] not in (older['source'], newer['source']) or older['source'] != newer['source']: excluded += 1; continue  # 排除训练泄漏和不一致的画面来源。
        if test['id'] not in previous or test['id'] not in current: continue  # 未经两轮实际计算的照片不参与准确率比较。
        before, after = previous[test['id']], current[test['id']]  # 读取同一张照片在两轮模型上的真实答案。
        rows.append(dict(id=test['id'], label=test['label'], before=before['predicted'], after=after['predicted'], before_scores=before['scores'], after_scores=after['scores']))  # 保留逐张证据而不仅给出汇总百分比。
    return dict(status='ready' if rows else 'empty', common=len(rows), excluded=excluded, before=sum(row['before'] == row['label'] for row in rows), after=sum(row['after'] == row['label'] for row in rows), rows=rows, coverage=len({row['label'] for row in rows}), before_id=older['id'], after_id=newer['id'])  # 三类覆盖不足时仍如实显示已测范围。

def check_result(test, scores):  # 规范保存一次真实模型输出与固定照片之间的联系。
    values = [float(value) for value in scores]  # 将数值库类型转换成可持久化的普通小数。
    return dict(id=test['id'], predicted=max(range(3), key=values.__getitem__), scores=values)  # 预测始终由得分最大值决定。

def validate_learning(payload, decode_image):  # 验证可选历史字段并兼容没有历史的旧课堂文件。
    if payload is None: return [], []  # 原有项目文件仍可直接载入。
    if not isinstance(payload, dict): raise ValueError('训练历史格式无效')  # 历史必须采用结构化对象。
    raw_tests, raw_runs = payload.get('tests', []), payload.get('runs', [])  # 只读取本版本支持的两种记录。
    if not isinstance(raw_tests, list) or len(raw_tests) > 60 or not isinstance(raw_runs, list) or len(raw_runs) > 6: raise ValueError('训练历史超出保留范围')  # 限制导入的图片和结果规模。
    tests, runs, identities = [], [], set()  # 分别累积已验证照片、轮次与不重复标识。
    for test in raw_tests:  # 对每张检验照片执行与普通拍照相同的验证。
        if not isinstance(test, dict) or type(test.get('label')) is not int or test['label'] not in range(3) or test.get('source') not in ['camera', 'simulation']: raise ValueError('固定检验照片无效')  # 保留独立的人工标签和来源。
        image, _ = decode_image(test.get('image')); digest = fingerprint(image)  # 重新计算实际像素标识而不信任导入的摘要。
        identity = test.get('id')  # 历史得分使用该稳定标识关联照片。
        if not isinstance(identity, str) or not re.fullmatch(r'[a-f0-9]{32}', identity) or identity in identities: raise ValueError('固定检验照片标识无效')  # 防止重复身份把两张照片错误对应。
        if any(item['digest'] == digest for item in tests): raise ValueError('固定检验集中有重复照片')  # 一张照片在检验集中只计一次。
        identities.add(identity); tests.append(dict(id=identity, image=image, digest=digest, label=test['label'], source=test['source']))  # 只保留支持的已验证字段。
    def numbers(value, length, integers=False):  # 验证固定长度的计数和得分数组。
        valid = isinstance(value, list) and len(value) == length and all(type(item) is int and item >= 0 if integers else type(item) in (int, float) and math.isfinite(item) and item >= 0 for item in value)  # 排除布尔值、负数和非有限数字。
        if not valid: raise ValueError('训练历史数值无效')  # 不发布无法解释的历史测量。
        return list(value)  # 与输入对象解耦后再保存。
    run_ids = set()  # 防止多个完成轮次共用身份。
    for run in raw_runs:  # 历史只恢复结果摘要，不恢复模型权重。
        if not isinstance(run, dict) or not isinstance(run.get('id'), str) or not re.fullmatch(r'[a-f0-9]{32}', run['id']) or run['id'] in run_ids: raise ValueError('训练轮次标识无效')  # 校验轮次关联键。
        counts, unique_counts = numbers(run.get('counts'), 3, True), numbers(run.get('unique_counts'), 3, True)  # 验证每类样本数量。
        if any(total > 120 or unique > total for total, unique in zip(counts, unique_counts)): raise ValueError('训练历史样本数量无效')  # 计数须满足课堂容量和去重约束。
        curve = numbers(run.get('curve'), 80); digests = run.get('training_digests'); names = run.get('names')  # 要求完成轮次具备全部八十次实测误差。
        if not isinstance(digests, list) or len(digests) > 360 or any(not isinstance(item, str) or not re.fullmatch(r'[a-f0-9]{64}', item) for item in digests): raise ValueError('训练数据版本无效')  # 历史像素标识仅用于排除训练泄漏。
        if not isinstance(names, list) or len(names) != 3 or any(not isinstance(name, str) or not name.strip() or len(name) > 12 for name in names): raise ValueError('训练历史类别无效')  # 保留当时的三类命名。
        if run.get('source') not in ['camera', 'simulation'] or not isinstance(run.get('dataset'), str) or not re.fullmatch(r'[a-f0-9]{64}', run['dataset']): raise ValueError('训练历史来源无效')  # 固定检验不得跨不一致来源比较。
        fit, raw_checks = run.get('fit'), run.get('checks', [])  # 读取每类拟合统计和独立检验结果。
        if not isinstance(fit, list) or len(fit) != 3 or not isinstance(raw_checks, list) or len(raw_checks) > 60: raise ValueError('训练历史结果无效')  # 限制汇总与逐图记录的结构。
        clean_fit = []  # 从原始计数重新构造各类测量。
        for label, item in enumerate(fit):  # 检查每类答对数不超过实际训练张数。
            if not isinstance(item, dict) or type(item.get('correct')) is not int or not 0 <= item['correct'] <= counts[label] or item.get('total') != counts[label]: raise ValueError('训练历史分类统计无效')  # 不接受自相矛盾的历史分数。
            clean_fit.append(dict(label=label, correct=item['correct'], total=counts[label]))  # 保存按固定类别顺序排列的统计。
            if 'loss' in item: clean_fit[-1]['loss'] = numbers([item['loss']], 1)[0]  # 保留新增的真实分类损失并验证有限非负数，旧课堂缺字段时继续兼容。
        checks, seen_checks = [], set()  # 一个轮次对同一检验照片只保留一个结果。
        for item in raw_checks:  # 所有已导入得分仍须关联已保存的实际照片。
            if not isinstance(item, dict) or item.get('id') not in identities or item['id'] in seen_checks: raise ValueError('训练历史检验关联无效')  # 防止孤立结果或重复计数。
            scores = numbers(item.get('scores'), 3)  # 验证三类输出值都是有限数。
            if any(score > 1 for score in scores) or abs(sum(scores) - 1) > .02: raise ValueError('训练历史得分无效')  # 模型输出须保持概率范围和归一化总量。
            checks.append(check_result(item, scores)); seen_checks.add(item['id'])  # 由得分重新求答案，避免信任另填的预测标签。
        created = run.get('created')  # 保留完成时间供历史视图显示。
        if type(created) not in (int, float) or not math.isfinite(created) or created < 0: raise ValueError('训练历史时间无效')  # 防止异常日期破坏结果窗口。
        runs.append(dict(id=run['id'], created=created, dataset=run['dataset'], names=names, source=run['source'], counts=counts, unique_counts=unique_counts, training_digests=list(digests), curve=curve, fit=clean_fit, correct=sum(item['correct'] for item in clean_fit), total=sum(counts), checks=checks)); run_ids.add(run['id'])  # 发布经过校验的只读历史摘要。
    return runs, tests  # 整份导入验证通过后才交给课堂状态发布。
