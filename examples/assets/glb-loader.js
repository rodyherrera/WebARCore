/**
 * GLBModelLoader - Utility class for loading and managing GLB/GLTF models
 * Provides helper methods for loading, scaling, and positioning 3D models
 */
export class GLBModelLoader {
    constructor(THREE, GLTFLoader) {
        this.THREE = THREE;
        this.loader = new GLTFLoader();
    }

    /**
     * Load a GLB model from a file
     * @param {File} file - The GLB/GLTF file to load
     * @param {Function} onProgress - Optional progress callback
     * @returns {Promise<Object>} - Promise resolving to the loaded GLTF object
     */
    loadFromFile(file, onProgress = null) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                const arrayBuffer = event.target.result;
                
                this.loader.parse(
                    arrayBuffer, 
                    '', 
                    (gltf) => resolve(gltf),
                    (error) => reject(error)
                );
            };

            reader.onerror = (error) => reject(error);
            reader.readAsArrayBuffer(file);
        });
    }

    /**
     * Load a GLB model from a URL
     * @param {string} url - The URL of the GLB/GLTF file
     * @param {Function} onProgress - Optional progress callback
     * @returns {Promise<Object>} - Promise resolving to the loaded GLTF object
     */
    loadFromURL(url, onProgress = null) {
        return new Promise((resolve, reject) => {
            this.loader.load(
                url,
                (gltf) => resolve(gltf),
                onProgress,
                (error) => reject(error)
            );
        });
    }

    /**
     * Center a model and normalize its size
     * @param {THREE.Object3D} model - The model to process
     * @param {number} targetSize - The desired size for the model
     * @returns {Object} - Object containing centering offset and scale factor
     */
    centerAndNormalize(model, targetSize = 1) {
        const box = new this.THREE.Box3().setFromObject(model);
        const center = box.getCenter(new this.THREE.Vector3());
        const size = box.getSize(new this.THREE.Vector3());
        
        // Calculate scale factor to fit within target size
        const maxDim = Math.max(size.x, size.y, size.z);
        const scaleFactor = targetSize / maxDim;
        
        return {
            center,
            size,
            scaleFactor,
            boundingBox: box
        };
    }

    /**
     * Apply automatic centering and scaling to a model
     * @param {THREE.Object3D} model - The model to transform
     * @param {number} targetSize - The desired size
     * @param {THREE.Vector3} position - Optional position offset
     */
    autoFit(model, targetSize = 1, position = null) {
        const { center, scaleFactor } = this.centerAndNormalize(model, targetSize);
        
        // Apply scale
        model.scale.set(scaleFactor, scaleFactor, scaleFactor);
        
        // Center the model
        model.position.sub(center.multiplyScalar(scaleFactor));
        
        // Apply custom position if provided
        if (position) {
            model.position.add(position);
        }
        
        return model;
    }

    /**
     * Get model information
     * @param {THREE.Object3D} model - The model to analyze
     * @returns {Object} - Information about the model
     */
    getModelInfo(model) {
        const box = new this.THREE.Box3().setFromObject(model);
        const size = box.getSize(new this.THREE.Vector3());
        const center = box.getCenter(new this.THREE.Vector3());
        
        let meshCount = 0;
        let materialCount = 0;
        let vertexCount = 0;
        
        model.traverse((child) => {
            if (child.isMesh) {
                meshCount++;
                if (child.geometry) {
                    vertexCount += child.geometry.attributes.position?.count || 0;
                }
                if (child.material) {
                    materialCount++;
                }
            }
        });
        
        return {
            size: {
                x: size.x.toFixed(2),
                y: size.y.toFixed(2),
                z: size.z.toFixed(2)
            },
            center: {
                x: center.x.toFixed(2),
                y: center.y.toFixed(2),
                z: center.z.toFixed(2)
            },
            meshCount,
            materialCount,
            vertexCount
        };
    }

    /**
     * Enable shadows for all meshes in a model
     * @param {THREE.Object3D} model - The model to process
     * @param {boolean} castShadow - Whether the model should cast shadows
     * @param {boolean} receiveShadow - Whether the model should receive shadows
     */
    enableShadows(model, castShadow = true, receiveShadow = true) {
        model.traverse((child) => {
            if (child.isMesh) {
                child.castShadow = castShadow;
                child.receiveShadow = receiveShadow;
            }
        });
    }

    /**
     * Create a clone of a loaded model
     * @param {THREE.Object3D} model - The model to clone
     * @returns {THREE.Object3D} - Cloned model
     */
    clone(model) {
        return model.clone(true);
    }
}
