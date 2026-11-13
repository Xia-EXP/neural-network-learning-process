/* 精修版的渐进增强层：只做原版脚本没有覆盖、但无障碍与可理解性必需的那几件事。
   1) 把进度轨的视觉状态同步成 aria-current，读屏用户才知道"现在在第几步"。
   2) 给动态生成的样本缩略图补 alt，避免一堆没有名字的图。
   3) 放大训练视图按钮补 aria-pressed；放大状态下按 Esc 可退出（键盘用户没有"再点一次"的直觉）。
   所有逻辑都做幂等判断，和原版脚本并发执行也不会打架。 */
(() => {
  'use strict';

  const body = document.body;

  /* ---- 1. 进度轨语义 ---- */
  const journey = document.getElementById('journey');
  if (journey) {
    const syncJourney = () => {
      journey.querySelectorAll('li').forEach((item) => {
        if (item.classList.contains('current')) item.setAttribute('aria-current', 'step');
        else item.removeAttribute('aria-current');
      });
    };
    syncJourney();
    // effects.js 在每次状态刷新时重写 li 的类名，所以监听属性而不是只跑一次。
    new MutationObserver(syncJourney).observe(journey, { subtree: true, attributes: true, attributeFilter: ['class'] });
  }

  /* ---- 2. 样本缩略图 alt ---- */
  const classList = document.getElementById('classList');
  if (classList) {
    const labelImages = () => {
      classList.querySelectorAll('.class-record').forEach((record) => {
        const name = record.querySelector('.class-name')?.textContent?.trim() || '类别';
        record.querySelectorAll('.thumbnails img').forEach((img, index) => {
          if (!img.getAttribute('alt')) img.setAttribute('alt', `${name} 的第 ${index + 1} 张训练样本`);
        });
        const capture = record.querySelector('.capture-button');
        if (capture && !capture.getAttribute('aria-label')) {
          capture.setAttribute('aria-label', `为「${name}」采集一张样本`);
        }
        const remove = record.querySelector('.delete-class');
        if (remove && !remove.getAttribute('aria-label')) {
          remove.setAttribute('aria-label', `清空「${name}」的样本`);
        }
      });
    };
    labelImages();
    // 缩略图由 app.js 反复重建，所以监听子节点变化而不是只跑一次。
    new MutationObserver(labelImages).observe(classList, { childList: true, subtree: true });
  }

  /* ---- 3. 训练视图放大 ---- */
  const focusButton = document.getElementById('trainingFocus');
  const syncFocusState = () => {
    const focused = body.classList.contains('training-focus');
    if (!focusButton) return;
    focusButton.setAttribute('aria-pressed', String(focused));
    focusButton.setAttribute('title', focused ? '退出放大视图（Esc）' : '放大训练视图');
    focusButton.setAttribute('aria-label', focused ? '退出放大视图' : '放大训练视图');
    const icon = focusButton.querySelector('[data-lucide]');
    if (icon) icon.setAttribute('data-lucide', focused ? 'minimize' : 'expand');
  };
  if (focusButton) {
    syncFocusState();
    // 原版通过切换 body 的类名实现放大，这里跟随同一个信号。
    new MutationObserver(syncFocusState).observe(body, { attributes: true, attributeFilter: ['class'] });
    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape' || !body.classList.contains('training-focus')) return;
      // 弹窗打开时 Esc 属于弹窗，交给浏览器处理。
      if (document.querySelector('dialog[open]')) return;
      event.preventDefault();
      focusButton.click();
      focusButton.focus();
    });
  }

  /* ---- 4. 04/05 浮层舞台 ----
     桌面端整页一屏，所以「学习成果 / 检验」两段平时不占位；点进度轨上的
     「懂」「考」才全屏展开，Esc 或右上角关闭按钮回到三栏工作台。
     窄屏（≤1100px）时这两段本来就在文档流里，浮层逻辑自动失效。 */
  const overlayTargets = {
    understand: document.getElementById('conceptSection'),
    test: document.getElementById('outcomeSection'),
  };
  const wideScreen = window.matchMedia('(min-width: 1101px)');

  const closeButton = document.createElement('button');
  closeButton.type = 'button';
  closeButton.className = 'ux-overlay-close';
  closeButton.hidden = true;
  closeButton.setAttribute('aria-label', '关闭，回到工作台');
  closeButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>';
  body.append(closeButton);

  let lastFocus = null;

  const closeOverlay = (restoreFocus = true) => {
    const open = body.dataset.uxOverlay;
    if (!open) return;
    const target = overlayTargets[open];
    if (target) {
      target.removeAttribute('role');
      target.removeAttribute('aria-modal');
    }
    body.classList.remove('ux-overlay');
    delete body.dataset.uxOverlay;
    body.appendChild(closeButton);
    closeButton.hidden = true;
    if (restoreFocus && lastFocus && document.contains(lastFocus)) lastFocus.focus({ preventScroll: true });
    lastFocus = null;
  };

  const openOverlay = (kind) => {
    const target = overlayTargets[kind];
    if (!target || !wideScreen.matches) return;
    if (body.dataset.uxOverlay === kind) return;
    if (!body.dataset.uxOverlay) lastFocus = document.activeElement;
    body.classList.add('ux-overlay');
    body.dataset.uxOverlay = kind;
    Object.values(overlayTargets).forEach((section) => {
      if (!section) return;
      section.removeAttribute('role');
      section.removeAttribute('aria-modal');
    });
    target.setAttribute('role', 'dialog');
    target.setAttribute('aria-modal', 'true');
    // 「颜色拓展实验」在原版是默认收起的可选实验；进入浮层时它才是主角，先展开。
    const experiment = target.querySelector('#colorExperiment');
    if (experiment && !experiment.open) experiment.open = true;
    target.appendChild(closeButton);
    closeButton.hidden = false;
    closeButton.focus({ preventScroll: true });
    if (typeof window.icons === 'function') window.icons();
  };

  closeButton.addEventListener('click', () => closeOverlay());

  journey?.addEventListener('click', (event) => {
    const item = event.target.closest('li[data-step]');
    if (!item) return;
    const step = item.dataset.step;
    if (step === 'understand' || step === 'test') openOverlay(step);
    else closeOverlay(false);
  });

  // 训练结束后原版会浮出「看看 AI 学会了什么」；04 现在是浮层，接住它。
  document.addEventListener('click', (event) => {
    if (event.target.closest('.jump-pill')) openOverlay('understand');
    if (event.target.closest('#processShowResult')) openOverlay('understand');
    if (event.target.closest('#sampleResultExam')) openOverlay('test');
  });

  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !body.dataset.uxOverlay) return;
    if (document.querySelector('dialog[open]')) return;
    event.preventDefault();
    closeOverlay();
  });

  // 从浮层状态拉宽/缩窄窗口时保持状态不打架：变窄就交还给文档流。
  wideScreen.addEventListener('change', (event) => {
    if (!event.matches) closeOverlay(false);
  });

  /* ---- 5. 让页签标题反映当前进度，切回标签页时不用重新找位置 ---- */
  const baseTitle = document.title;
  const stageLabels = { teach: '样本标定', learn: '观察训练', understand: '颜色地图', test: '检验结果' };
  if (journey) {
    const syncTitle = () => {
      const step = journey.querySelector('li.current')?.dataset.step;
      document.title = step && stageLabels[step] ? `${baseTitle} · ${stageLabels[step]}` : baseTitle;
    };
    syncTitle();
    new MutationObserver(syncTitle).observe(journey, { subtree: true, attributes: true, attributeFilter: ['class'] });
  }
})();

