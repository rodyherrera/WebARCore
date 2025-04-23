import { Stats } from '../assets/stats.js';
import { WebARCore } from '../assets/webarcore.js';
import { ARPoseRendererView } from '../assets/view.js';
import { Camera, runPerFrameLoop, scaleToCover } from '../assets/utils.js';

const main = () => {
    const config = {
        video: {
            facingMode: { ideal: 'environment' },
            aspectRatio: 16 / 9,
            width: { ideal: 1280 }
        },
        audio: false
    };

    const container = document.getElementById('container');
    const viewContainer = document.createElement('div');
    const canvas = document.createElement('canvas');
    const overlay = document.getElementById('overlay');
    const startBtn = document.getElementById('startBtn');
    const splash = document.getElementById('splash');
    const splashFadeTime = 800;

    splash.style.transition = `opacity ${ splashFadeTime / 1000 }s ease`;
    splash.style.opacity = 0;

    const demo = async (media) => {
        const video = media.element;
        const size = scaleToCover(video.videoWidth, video.videoHeight, container.clientWidth, container.clientHeight);
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        video.style.width = size.width + 'px';
        video.style.height = size.height + 'px';

        const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
        const webarcore = await WebARCore.Initialize(canvas.width, canvas.height);
        const view = new ARPoseRendererView(viewContainer, canvas.width, canvas.height);
        
        Stats.addTimer('total');
        Stats.addTimer('video');
        Stats.addTimer('slam');
        
        container.appendChild(canvas);
        container.appendChild(viewContainer);
      
        document.body.appendChild(Stats.uiElement);
        document.body.addEventListener('click', () => webarcore.reset(), false);
        
        runPerFrameLoop(() => {
            Stats.startFrame();
            Stats.startTimer('total');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if(!document['hidden']){
                Stats.startTimer('video');
                ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, size.x, size.y, size.width, size.height);
                const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
                Stats.stopTimer('video');
                Stats.startTimer('slam');
                const pose = webarcore.findCameraPose(frame);
                Stats.stopTimer('slam');
                if(pose){
                    view.updateCameraPose(pose);
                }else{
                    view.lostCamera();
                    const dots = webarcore.getFramePoints();
                    for(const { x, y } of dots){
                        ctx.fillStyle = 'white';
                        ctx.fillRect(x, y, 2, 2);
                    }
                }
            }

            Stats.stopTimer('total');
            Stats.renderUI();

            return true;
        }, 30);
    };

    setTimeout(() => {
        splash.remove();
        startBtn.addEventListener('click', () => {
            overlay.remove();
            Camera.Initialize(config)
                .then((media) => demo(media))
                .catch((error) => alert('Camera ' + error));
        }, { once: true });
    }, splashFadeTime);
};

window.addEventListener('load', main);