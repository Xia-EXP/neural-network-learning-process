'use strict'; // Keep the GM92 portrait big-screen arrangement separate from the shared lesson logic.
if(gm92Panel){ // Apply only to the operable ?screen=gm92 entry; ?remote=1 and ?display=1 keep their own behavior.
    document.body.classList.add('gm92');document.title='三格学习台 · GM92 触控课堂'; // Select the portrait layout and name the page for the browser task list.
    const params=new URLSearchParams(location.search),reserve=Math.max(0,Math.min(200,Number(params.get('reserve'))||0)); // Read an optional bottom reserve (CSS px) for a system bar that overlays the page in immersive mode.
    document.documentElement.style.setProperty('--sys-bottom',reserve+'px'); // Lift the step bar above that reserved strip without relying on env() support.
    function fitHeight(){const height=Math.round(window.visualViewport?.height||window.innerHeight);document.documentElement.style.setProperty('--app-h',height+'px');fitViewfinder();} // Measure the real usable height instead of trusting 100vh/100dvh on an unverified WebView.
    function fitViewfinder(){const finder=document.querySelector('.infer-stage .viewfinder'),stage=document.querySelector('.infer-stage');if(!finder||!stage||!stage.clientWidth)return;const room=stage.clientWidth-parseFloat(getComputedStyle(stage).paddingLeft)-parseFloat(getComputedStyle(stage).paddingRight),share=document.body.dataset.lessonView==='result'?.30:.34,limit=(window.visualViewport?.height||innerHeight)*share*4/3,width=Math.floor(Math.max(200,Math.min(room,limit)));document.documentElement.style.setProperty('--vf-w',width+'px');} // Size the shared picture in pixels so the 4:3 frame does not depend on aspect-ratio support and leaves the buttons in the lower half.
    window.addEventListener('resize',fitHeight);window.visualViewport?.addEventListener('resize',fitHeight); // Re-measure after rotation, browser-bar changes, or soft-keyboard display.
    new ResizeObserver(fitViewfinder).observe(document.querySelector('.infer-stage')); // Refit the picture when its tab becomes visible or its column changes width.
    const tools=document.querySelector('.project-tools'),chip=$('connectionState');chip.classList.add('gm92-link');tools.prepend(chip); // Show the measured camera-sharing state in the slim header instead of the hidden desktop footer.
    const train=document.createElement('button');train.id='gm92Train';train.className='primary-button';train.innerHTML='<i data-lucide="play"></i><span>开始训练</span>';train.onclick=()=>trainModel(); // Put the real training command on the teach page; trainModel itself refuses whenever #train is disabled.
    const note=document.createElement('div');note.id='gm92TrainNote';note.className='gm92-note'; // Explain the actual prerequisite next to the command.
    const box=document.createElement('div');box.className='gm92-train-box';box.append(train,note);document.querySelector('.touch-label-section').after(box); // Place the command directly below the three capture buttons, within reach of students.
    const base=window.renderRemote; // Keep the shared touch-panel rendering and add GM92 details after it.
    window.renderRemote=function(){ // Mirror real state onto GM92-only controls on every state update.
        base(); // Update badges, counts, and automatic step changes exactly as on the touch panel.
        const real=$('train'),need=state.counts.map((c,i)=>c<3?state.names[i]+' 还差 '+(3-c)+' 张':'').filter(Boolean); // Read the genuine training command state and missing samples.
        train.disabled=real.disabled;train.querySelector('span').textContent=real.querySelector('span').textContent; // Never enable or relabel the GM92 command differently from the real one.
        note.textContent=state.training?'正在训练，请看“学”页':need.length?'每类至少 3 张才能训练：'+need.join('，'):state.ready?'已训练完成；补充样本后可重新训练':'样本已够，可以开始训练'; // State exactly why training is or is not available.
        const live=sourceReady();document.body.classList.toggle('gm92-offline',!live);chip.textContent=live?'● 电脑画面已连接':'○ 无电脑画面'; // Report the measured sharing state; capture and examination stay disabled while it is offline.
        if(document.body.dataset.lessonView==='label')$('remoteHint').textContent=!live?'先在电脑上连接摄像头并保持页面打开，这里才能采集':state.counts.every(c=>c>=3)?(state.training?'正在训练…':'每类都够 3 张了，点下面“开始训练”'):'对准样品，点颜色按钮给它起名字（每类至少 3 张）'; // Point to the on-page training command instead of the tab bar.
    };
    document.querySelectorAll('button[data-lesson-view]').forEach(button=>button.addEventListener('click',()=>requestAnimationFrame(fitViewfinder))); // Resize the picture only when an actual navigation button is tapped.
    fitHeight();window.renderRemote();icons(); // Draw the initial GM92 state with real measurements.
}
