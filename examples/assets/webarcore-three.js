/**
 * Example usage:
 * const webarcore = await WebARCore.Initialize(...);
 * const applyPose = WebARCoreConnectorTHREE.Initialize(THREE);
 * const renderer = mew THREE.WebGLRenderer(...);
 * const camera = new THREE.PerspectiveCamera(...);
 * const scene = new THREE.Scene();
 * 
 * const loop () => {
 *     const imageData = ctx.getImageData(...);
 *     const pose = webarcore.findCameraPose(imagedata);
 *     if(pose) applyPose(pose, camera.quaternion, camera.position);
 *     renderer.render(this.scene, this.camera);
 * }
 */

class WebARCoreConnectorTHREE{
    static Initialize(THREE){
        return (pose, rotationQuaternion, translationVector) => {
            const matrix = new THREE.Matrix4().fromArray(pose);
            const quaternion = new THREE.Quaternion().setFromRotationMatrix(matrix);
            const translation = new THREE.Vector3(pose[12], pose[13], pose[14]);
            if(rotationQuaternion !== null){
                rotationQuaternion.set(-quaternion.x, quaternion.y, quaternion.z, quaternion.w);
            }
            if(translationVector !== null){
                translationVector.set(translation.x, -translation.y, -translation.z);
            }
        };
    }
}

export { WebARCoreConnectorTHREE }