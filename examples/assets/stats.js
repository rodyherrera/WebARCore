const singletonEnforcer = Symbol('Singleton Enforcer');

class StatsTimer{
    constructor(bufferLength = 30){
        this.startTime = 0;
        this.elapsedTime = 0;
        this.history = new Array(bufferLength).fill(0);
        this.currentIndex = 0;
    }

    start(){
        this.startTime = performance.now();
    }

    stop(){
        this.elapsedTime = Math.round(performance.now() - this.startTime);
        this.history[this.currentIndex] = this.elapsedTime;
        this.currentIndex = (this.currentIndex + 1) % this.history.length;
    }

    reset(){
        this.elapsedTime = 0;
    }

    getElapsedTime(){
        return this.elapsedTime;
    }

    getAverageElapsedTime(){
        const sum = this.history.reduce((acc, value) => acc + (value / this.history.length), 0);
        return Math.round(sum);
    }
}

class CircularBuffer{
    constructor(capacity){
        this.head = 0;
        this.tail = -1;
        this.count = 0;
        this.capacity = capacity;
        this.data = new Int32Array(capacity);
    }

    push(value){
        if(this.count < this.capacity){
            this.tail++;
            this.data[this.tail] = value;
            this.count++;
        }else{
            this.tail = (this.tail + 1) % this.capacity;
            this.head = (this.head + 1) % this.capacity;
            this.data[this.tail] = value;
        }
    }

    get(index){
        return this.data[(this.head + index) % this.capacity];
    }

    size(){
        return this.count;
    }
}

class StatsManager{
    constructor(enforcer){
        if(enforcer !== singletonEnforcer){
            throw new Error('StatsManager is a singleton. Use the exported instance.');
        }

        if(!StatsManager.instance){
            StatsManager.instance = this;
        }

        this.frameCount = 0;
        this.framesPerSecond = 0;
        this.timers = [];
        this.frameTimer = new StatsTimer();
        this.frameTimeBuffer = new CircularBuffer(16);
        this.fpsHistory = new Array(50).fill(0);

        this.memoryInfoSupported = perfomance?.memory?.totalJSHeapSize !== undefined;
        this.memoryInBytes = 0;
        this.bytesToMegabytes = 1_000_000;

        this.uiElement = document.createElement('div');
        this.uiElement.style = `
            background:rgba(255,255,255,0.8); 
            position:absolute;
            top:0px; 
            left:0px; 
            display:block; 
            min-width:80px; 
            color:black; 
            font: 10px Arial, sans-serif;
            padding:5px;
        `;

        return StatsManager.instance;
    }

    addTimer(name){
        this.timers.push([name, new StatsTimer()]);
    }

    startFrame(){
        this.frameCount++;
        this.timers.forEach(([_, timer]) => timer.reset());

        if(this.frameCount > 0){
            this.frameTimer.stop();
            this.frameTimeBuffer.push(this.frameTimer.getElapsedTime());

            const totalFrames = this.frameTimeBuffer.size();
            let totalElapsed = 0;

            for(let i = 0; i < totalFrames; i++){
                totalElapsed += this.frameTimeBuffer.get(i);
            }

            this.framesPerSecond = (totalFrames / totalElapsed) * 100;
            this.fpsHistory[this.frameCount % this.fpsHistory.length] = Math.round(this.framesPerSecond);

            this.frameTimer.start();
        }

        if(this.memoryInfoSupported){
            this.memoryBytes = perfomance.memory.usedJSHeapSize;
        }
    }

    getTimer(name){
        return this.timers.find(([timerName]) => timerName === name);
    }

    startTimer(name){
        this.getTimer(name)[1].start();
    }

    stopTimer(name){
        this.getTimer(name)[1].stop();
    }

    getInfoString(){
        let info = `FPS: ${Math.round(this.framesPerSecond)} (${Math.min(...this.fpsHistory)} - ${Math.max(...this.fpsHistory)})`;
        this.timers.forEach(([name, timer]) => {
            info += `\n${name} : ${timer.getElapsedTime()}ms`;
        });
        if(this.memoryInfoSupported){
            info += `\nMemory: ${(this.memoryInBytes / this.bytesToMegabytes).toFixed(2)}MB`;
        }
        return info;
    }

    renderUI(extraInfo = null){
        let html = `<b>FPS: ${Math.round(this.framesPerSecond)} (${Math.min(...this.fpsHistory)} - ${Math.max(...this.fpsHistory)})</b>`;

        this.timers.forEach(([name, timer]) => {
            html += `<br/>${name} : ${timer.getAverageElapsedTime()}ms`;
        });

        if(this.memoryInfoSupported){
            html += `<br/>Memory: ${(this.memoryInBytes / this.bytesToMegabytes).toFixed(2)}MB`;
        }

        if(extraInfo){
            html += `<br/>${extraInfo}`;
        }

        this.uiElement.innerHTML = html;
    }
}

const Stats = new StatsManager(singletonEnforcer);

export { Stats };