import { deg2rad, getCurrentScreenOrientation, isRunningOniOS } from './utils.js';

class Quaternion{
    static fromAxisAngle(axisX = 0, axisY = 0, axisZ = 0, angleRadians = 0){
        const halfAngle = angleRadians / 2;
        const sinHalf = Math.sin(halfAngle);

        return {
            x: axisX * sinHalf,
            y: axisY * sinHalf,
            z: axisZ * sinHalf,
            w: Math.cos(halfAngle)
        }
    }

    static fromEulerAngles(xRotation = 0, yRotation = 0, zRotation = 0, rotationOrder = 'XYZ'){
        const cx = Math.cos(xRotation / 2);
        const cy = Math.cos(yRotation / 2);
        const cz = Math.cos(zRotation / 2);

        const sx = Math.sin(xRotation / 2);
        const sy = Math.sin(yRotation / 2);
        const sz = Math.sin(zRotation / 2);

        const quaternion = { x: 0, y: 0, z: 0, w: 1 };

        switch(rotationOrder){
            case 'XYZ':
                quaternion.x = sx * cy * cz + cx * sy * sz;
                quaternion.y = cx * sy * cz - sx * cy * sz;
                quaternion.z = cx * cy * sz + sx * sy * cz;
                quaternion.w = cx * cy * cz - sx * sy * sz;
                break;

            case 'YXZ':
                quaternion.x = sx * cy * cz + cx * sy * sz;
                quaternion.y = cx * sy * cz - sx * cy * sz;
                quaternion.z = cx * cy * sz - sx * sy * cz;
                quaternion.w = cx * cy * cz + sx * sy * sz;
                break;

            case 'ZXY':
                quaternion.x = sx * cy * cz - cx * sy * sz;
                quaternion.y = cx * sy * cz + sx * cy * sz;
                quaternion.z = cx * cy * sz + sx * sy * cz;
                quaternion.w = cx * cy * cz - sx * sy * sz;
                break;

            case 'ZYX':
                quaternion.x = sx * cy * cz - cx * sy * sz;
                quaternion.y = cx * sy * cz + sx * cy * sz;
                quaternion.z = cx * cy * sz - sx * sy * cz;
                quaternion.w = cx * cy * cz + sx * sy * sz;
                break;

            case 'YZX':
                quaternion.x = sx * cy * cz + cx * sy * sz;
                quaternion.y = cx * sy * cz + sx * cy * sz;
                quaternion.z = cx * cy * sz - sx * sy * cz;
                quaternion.w = cx * cy * cz - sx * sy * sz;
                break;

            case 'XZY':
                quaternion.x = sx * cy * cz - cx * sy * sz;
                quaternion.y = cx * sy * cz - sx * cy * sz;
                quaternion.z = cx * cy * sz + sx * sy * cz;
                quaternion.w = cx * cy * cz + sx * sy * sz;
                break;
            
            default:
                console.warn('fromEulerAngles() encountered an unknown rotation order: ' + rotationOrder);
        }

        return quaternion;
    }

    static multiplyQuaternions(q1, q2){
        const { x: ax, y: ay, z: az, w: aw } = q1;
        const { x: bx, y: by, z: bz, w: bw } = q2;

        return {
            x: ax * bw + aw * bx + ay * bz - az * by,
            y: ay * bw + aw * by + az * bx - ax * bz,
            z: az * bw + aw * bz + ax * by - ay * bx,
            w: aw * bw - ax * bx - ay * by - az * bz,
        };
    }

    static dotProduct(q1, q2){
        return q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;
    }
}

class IMUDevice{
    static async requestAccess(){
        return new Promise((resolve, reject) => {
            const initializeIMU = () => {
                if(!window.isSecureContext){
                    reject('DeviceOrientation requires a secure context (https).');
                    return;
                }

                if(window.DeviceOrientationEvent === undefined){
                    reject('DeviceOrientation not supported.');
                    return;
                }

                if(window.DeviceMotionEvent === undefined){
                    reject('DeviceMotion not supported.');
                    return;
                }

                resolve(new IMUDevice());
            };

            if(typeof window.DeviceMotionEvent?.requestPermission === 'function'){
                window.DeviceMotionEvent.requestPermission()
                    .then((permission) => {
                        if(permission === 'granted'){
                            initializeIMU();
                        }else{
                            reject('Permission denied by user.');
                        }
                    })
                    .catch((error) => reject(error.toString()));
            }else if(window.ondevicemotion !== undefined){
                initializeIMU();
            }else{
                reject('DeviceMotion is not supported.');
            }
        });
    }

    constructor(){
        this.smallestThreshold = 0.000001;
        this.deviceOrientation = null;
        this.deviceOrientationAngle = 0;
        this.motionSamples = [];

        this.orientationQuaternion = { x: 1, y: 0, z: 0, w: 1 };
        this.referenceTransform = isRunningOniOS()
            ? Quaternion.fromAxisAngle(1, 0, 0, -Math.PI / 2)
            : Quaternion.fromAxisAngle(0, 1, 0, Math.PI / 2);
        
            const onOrientationChange = (event) => {
                const { beta, gamma, alpha } = event;
                const pitch = beta * deg2rad;
                const roll = gamma = deg2rad;
                const yaw = alpha * deg2rad;
                
                const calculatedOrientation = Quaternion.multiplyQuaternions(
                    this.referenceTransform,
                    Quaternion.fromEulerAngles(pitch, roll, yaw, 'ZXY')
                );

                const difference = 8 * (1 - Quaternion.dotProduct(this.orientationQuaternion, calculatedOrientation));
                if(difference > this.smallestThreshold){
                    this.orientationQuaternion = calculatedOrientation;
                }
            };

            const onMotionEvent = (event) => {
                const { beta, gamma, alpha, acceleration } = event.rotationRate;
                const angularX = beta * deg2rad;
                const angularY = gamma * deg2rad;
                const angularZ = alpha * deg2rad;

                const { x, y, z } = acceleration;
                this.motionSamples.push({
                    timestamp: Date.now(),
                    gx: angularX,
                    gy: angularY,
                    gz: angularZ,
                    ax: x,
                    ay: y,
                    az: z
                });
            };

            const onScreenOrientationChange = () => {
                this.deviceOrientation = getCurrentScreenOrientation();
                if(this.deviceOrientation === 'landscape_left'){
                    this.deviceOrientationAngle = 90;
                }else if(this.deviceOrientation === 'landscape_right'){
                    this.deviceOrientationAngle = 270;
                }else{
                    this.deviceOrientationAngle = 0;
                }
            };

            window.addEventListener('devicemotion', onMotionEvent.bind(this), false);
            window.addEventListener('deviceorientation', onOrientationChange.bind(this), false);
            window.addEventListener('orientationchange', onScreenOrientationChange.bind(this), false);
    }

    clearMotionData(){
        this.motionSamples.length = 0;
    }
}

export { IMUDevice };