export const radiansToDegrees = 180.0 / Math.PI;
export const deg2rad = Math.PI / 180;

export const runPerFrameLoop = (frameCallback, targetFps = 30) => {
    const frameIntervalMs = Math.floor(1000 / targetFps);
    let lastFrameTime = performance.now();

    const tick = async () => {
        const currentTime = performance.now();
        const deltaTime = currentTime - lastFrameTime;
        if(deltaTime > frameIntervalMs){
            lastFrameTime = currentTime - (deltaTime % frameIntervalMs);
            const continueLoop = await frameCallback(currentTime);
            if(continueLoop === false) return;
        }
        requestAnimationFrame(tick);
    };
    
    requestAnimationFrame(tick);
}

export const isRunningOniOS = () => {
    // TODO: check this, is deprecated??
    return /iPad|iPhone|iPod/.test(navigator.platform);
};

export const isTouchDevice = () => {
    try{
        document.createEvent('TouchEvent');
        return true;
    }catch{
        return false;
    }
};

export const getCurrentScreenOrientation = () => {
    let orientationAngle = -1;
    if(window.screen && window.screen.orientation){
        orientationAngle = window.screen.orientation.angle;
    }else if('orientation' in window){
        // TODO: is deprecated this?
        orientationAngle = window.orientation;
    }

    switch(orientationAngle){
        case 0:
        case 180:
            return 'portrait';
        case 90:
            return 'landscape_left';
        case 270:
        case -90:
            return 'landscape_right';
        default:
            return 'unknown';
    }
};

export const scaleToCover = (srcWidth, srcHeight, dstWidth, dstHeight) => {
    const output = {};
    if(dstWidth / dstHeight > srcWidth / srcHeight){
        const scale = dstWidth / srcWidth;
        output.width = Math.floor(scale * srcWidth);
        output.height = Math.floor(scale * srcHeight);
        output.x = 0;
        output.y = Math.floor((dstHeight - output.height) * 0.5);
    }else{
        const scale = dstHeight / srcHeight;
        output.width = Math.floor(scale * srcWidth);
        output.height = Math.floor(scale * srcHeight);
        output.x = Math.floor((dstWidth - output.width) * 0.5);
        output.y = 0;
    }
    return output;
};

const createCanvasElement = (width, height) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
};

export class Camera{
    static async initialize(constraints = null){
        if('facingMode' in constraints && 'deviceId' in constraints){
            throw new Error('Cannot used "deviceId" and "facingMode" together.');
        }
        if('facingMode' in constraints && !['environment', 'user'].includes(constraints.facingMode)){
            throw new Error('Invalid "facingMode" value.');
        }
        
        const createVideoStream = (permissionStatus) => {
            return new Promise((resolve, reject) => {
                const handleSuccess = (stream) => {
                    const videoTrack = stream.getVideoTracks()[0];
                    if(!videoTrack){
                        reject(new Error('No video track available.'));
                        return;
                    }
                    const videoElement = document.createElement('video');
                    videoElement.setAttribute('autoplay', 'autoplay');
                    videoElement.setAttribute('playsinline', 'playsinline');
                    videoElement.setAttribute('webkit-playsinline', 'webkit-playsinline');
                    videoElement.srcObject = stream;

                    videoElement.onloadedmetadata = () => {
                        const settings = videoTrack.getSettings();
                        const expectedWidth = settings.width;
                        const expectedHeight = settings.height;
                        const actualWidth = videoElement.videoWidth;
                        const actualHeight = videoElement.videoHeight;

                        if(actualWidth !== expectedWidth || actualHeight !== expectedHeight){
                            console.warn(`Video size mismatch: expected ${expectedWidth}x${expectedHeight}, got ${actualWidth}x${actualHeight}`);
                        }

                        videoElement.style.width = `${actualWidth}px`;
                        videoElement.style.height = `${actualHeight}px`;
                        videoElement.width = actualWidth;
                        videoElement.height = actualHeight;
                        videoElement.play();

                        resolve(new Camera(videoElement));
                    };
                };

                const handleError = (error) => {
                    const messageMap = {
                        NotFoundError: 'Camera not found.',
                        DevicesNotFoundError: 'Camera not found.',
                        SourceUnavailableError: 'Camera busy.',
                        PermissionDeniedError: 'Permission denied.',
                        SecurityError: 'Permission denied.'
                    };
                    reject(new Error(`Camera error: ${messageMap[error.name] || 'Rejected.'}`));
                };

                if(permissionStatus?.state === 'denied'){
                    reject(new Error('Camera permission denied.'));
                    return;
                }

                navigator.mediaDevices.getUserMedia(constraints)
                    .then(handleSuccess)
                    .catch(handleError);
            });
        };

        if(navigator.permissions?.query){
            return navigator.permissions.query({ name: 'camera' })
                .then(createVideoStream)
                .catch(() => createVideoStream());
        }else{
            return createVideoStream();
        }
    }

    constructor(videoElement){
        this.element = videoElement;
        this.width = videoElement.videoWidth;
        this.height = videoElement.videoHeight;
        this._canvas = createCanvasElement(this.width, this.height);
        this._context = this._canvas.getContext('2d', { willReadFrequently: true });
    }

    getImageData(){
        this._context.clearRect(0, 0, this.width, this.height);
        this._context.drawImage(this.element, 0, 0, this.width, this.height);
        return this._context.getImageData(0, 0, this.width, this.height);
    }
}

export class Video{
    static async initialize(sourceUrl, timeoutMs = 8000){
        return new Promise((resolve, reject) => {
            const videoElement = document.createElement('video');
            videoElement.src = sourceUrl;
            videoElement.setAttribute('autoplay', 'autoplay');
            videoElement.setAttribute('playsinline', 'playsinline');
            videoElement.setAttribute('webkit-playsinline', 'webkit-playsinline');
            videoElement.autoplay = true;
            videoElement.muted = true;
            videoElement.loop = true;
            videoElement.load();

            const timeoutId = setTimeout(() => {
                reject(new Error(`Failed to load video: Timed out after ${timeoutMs}ms.`));
            }, timeoutMs);

            videoElement.onerror = () => {
                clearTimeout(timeoutId);
                reject(new Error('Failed to load video.'));
            };

            videoElement.onabort = () => {
                clearTimeout(timeoutId);
                reject(new Error('Failed to load video: Load aborted.'));
            };

            if(videoElement.readyState >= 4){
                clearTimeout(timeoutId);
                resolve(videoElement);
            }else{
                videoElement.oncanplaythrough = () => {
                    clearTimeout(timeoutId);
                    if(videoElement.videoWidth === 0 || videoElement.videoHeight === 0){
                        reject(new Error('Failed to load video: Invalid dimensions.'));
                    }else{
                        resolve(videoElement);
                    }
                };
            }
        }).then((videoElement) => {
            videoElement.onload = videoElement.onabort = videoElement.onerror = null;
            return new Video(videoElement);
        });
    }

    constructor(videoElement) {
        this.element = videoElement;
        this.width = videoElement.videoWidth;
        this.height = videoElement.videoHeight;

        this._canvas = createCanvasElement(this.width, this.height);
        this._context = this._canvas.getContext('2d', { willReadFrequently: true });

        this._lastTimestamp = -1;
        this._cachedImageData = null;
    }

    getImageData(){
        const currentTimestamp = this.element.currentTime;
        if(this._lastTimestamp !== currentTimestamp){
            this._lastTimestamp = currentTimestamp;
            this._cachedImageData = null;
        }

        if(this._cachedImageData === null){
            this._context.clearRect(0, 0, this.width, this.height);
            this._context.drawImage(this.element, 0, 0, this.width, this.height);
            this._cachedImageData = this._context.getImageData(0, 0, this.width, this.height);
        }

        return this._cachedImageData;
    }
}