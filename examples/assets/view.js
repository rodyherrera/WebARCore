import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/build/three.module.js';
import { OrbitControls } from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/loaders/GLTFLoader.js';
import { WebARCoreConnectorTHREE } from './webarcore-three.js';

export class ARPoseRendererView{
    constructor(container, width, height, x = 0, y = 0, z = -10, scale = 1.0){
        this.applyPose = WebARCoreConnectorTHREE.Initialize(THREE);
        
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setClearColor(0, 0);
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera.rotation.reorder('YXZ');
        this.camera.updateProjectionMatrix();

        // Create default object (icosahedron)
        this.defaultObject = new THREE.Mesh(
            new THREE.IcosahedronGeometry(1, 0),
            new THREE.MeshNormalMaterial({ flatShading: true })
        );
        
        this.object = this.defaultObject;
        this.object.scale.set(scale, scale, scale);
        this.object.position.set(x, y, z);
        this.object.visible = false;

        this.scene = new THREE.Scene();
        
        // Enhanced lighting for better model visibility
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8); // Increased intensity
        this.scene.add(ambientLight);
        
        const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0); // Increased intensity
        hemisphereLight.position.set(0, 20, 0);
        this.scene.add(hemisphereLight);
        
        // Add directional light for better depth perception
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 7.5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        // Add a point light that follows the camera
        this.cameraLight = new THREE.PointLight(0xffffff, 0.5, 100);
        this.camera.add(this.cameraLight);
        
        this.scene.add(this.camera);
        this.scene.add(this.object);

        // GLTFLoader for loading GLB models
        this.gltfLoader = new GLTFLoader();
        this.currentModel = null;
        this.baseScale = scale;
        this.basePosition = { x, y, z };

        container.appendChild(this.renderer.domElement);

        const render = () => {
            requestAnimationFrame(render.bind(this));
            this.renderer.render(this.scene, this.camera);
        };

        render();
    }

    loadGLBModel(file, onProgress = null){
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                const arrayBuffer = event.target.result;
                
                this.gltfLoader.parse(arrayBuffer, '', (gltf) => {
                    // Remove current model if exists
                    if(this.currentModel){
                        this.scene.remove(this.object);
                        this.object = null;
                    }

                    // Set up the new model
                    this.currentModel = gltf.scene;
                    this.object = this.currentModel;
                    
                    // Center the model
                    const box = new THREE.Box3().setFromObject(this.object);
                    const center = box.getCenter(new THREE.Vector3());
                    this.object.position.set(
                        this.basePosition.x - center.x,
                        this.basePosition.y - center.y,
                        this.basePosition.z - center.z
                    );

                    // Apply base scale
                    const size = box.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const scaleFactor = (this.baseScale * 2) / maxDim;
                    this.object.scale.set(scaleFactor, scaleFactor, scaleFactor);
                    
                    this.object.visible = false;
                    this.scene.add(this.object);
                    
                    console.log('GLB model loaded successfully');
                    resolve(gltf);
                }, 
                (error) => {
                    console.error('Error loading GLB model:', error);
                    reject(error);
                });
            };

            reader.onerror = (error) => {
                console.error('Error reading file:', error);
                reject(error);
            };

            reader.readAsArrayBuffer(file);
        });
    }

    loadGLBModelFromURL(url, onProgress = null){
        return new Promise((resolve, reject) => {
            this.gltfLoader.load(
                url,
                (gltf) => {
                    // Remove current model if exists
                    if(this.currentModel){
                        this.scene.remove(this.object);
                        this.object = null;
                    }

                    // Set up the new model
                    this.currentModel = gltf.scene;
                    this.object = this.currentModel;
                    
                    // Enable shadows and ensure materials are visible
                    this.object.traverse((child) => {
                        if(child.isMesh){
                            child.castShadow = true;
                            child.receiveShadow = true;
                            // Ensure material is visible
                            if(child.material){
                                child.material.side = THREE.DoubleSide;
                                // If material is too dark, add emissive
                                if(!child.material.emissive){
                                    child.material.emissive = new THREE.Color(0x222222);
                                }
                            }
                        }
                    });
                    
                    // Center the model
                    const box = new THREE.Box3().setFromObject(this.object);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());
                    
                    console.log('Model dimensions:', {
                        x: size.x.toFixed(2),
                        y: size.y.toFixed(2), 
                        z: size.z.toFixed(2)
                    });
                    
                    // Apply base scale - use larger multiplier for better visibility
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const scaleFactor = (this.baseScale * 4) / maxDim; // Increased from 2 to 4
                    this.object.scale.set(scaleFactor, scaleFactor, scaleFactor);
                    
                    // Position the model
                    this.object.position.set(
                        this.basePosition.x,
                        this.basePosition.y,
                        this.basePosition.z
                    );
                    
                    this.object.visible = false;
                    this.scene.add(this.object);
                    
                    console.log('GLB model loaded successfully from URL:', url);
                    console.log('Model position:', this.object.position);
                    console.log('Model scale:', scaleFactor.toFixed(2));
                    console.log('Model will be visible when tracking starts');
                    resolve(gltf);
                },
                (progress) => {
                    if(onProgress){
                        onProgress(progress);
                    }
                },
                (error) => {
                    const errorMsg = error.message || 'Unknown error loading model';
                    console.error('Error loading GLB model from URL:', url);
                    console.error('Error details:', errorMsg);
                    
                    // Provide helpful error message
                    if(errorMsg.includes('404') || errorMsg.includes('Not Found')){
                        reject(new Error(`Model file not found at: ${url}`));
                    } else if(errorMsg.includes('CORS')){
                        reject(new Error(`CORS error loading model from: ${url}`));
                    } else {
                        reject(new Error(`Failed to load model: ${errorMsg}`));
                    }
                }
            );
        });
    }

    resetToDefaultModel(){
        if(this.currentModel){
            this.scene.remove(this.object);
            this.currentModel = null;
        }

        this.object = this.defaultObject;
        this.object.position.set(this.basePosition.x, this.basePosition.y, this.basePosition.z);
        this.object.scale.set(this.baseScale, this.baseScale, this.baseScale);
        this.object.visible = false;
        
        if(!this.scene.children.includes(this.object)){
            this.scene.add(this.object);
        }
        
        console.log('Reset to default model');
    }

    setScale(scale){
        if(this.object){
            const currentScale = this.object.scale.x;
            const ratio = scale / this.baseScale;
            
            if(this.currentModel){
                // For GLB models, maintain the normalized scale
                const box = new THREE.Box3().setFromObject(this.currentModel);
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const scaleFactor = (scale * 2) / maxDim;
                this.object.scale.set(scaleFactor, scaleFactor, scaleFactor);
            } else {
                // For default model
                this.object.scale.set(scale, scale, scale);
            }
            
            this.baseScale = scale;
        }
    }

    updateCameraPose(pose){
        this.applyPose(pose, this.camera.quaternion, this.camera.position);
        
        if(!this.object.visible){
            console.log('🎯 Model is now visible! Camera tracking active.');
            console.log('Model info:', {
                position: {
                    x: this.object.position.x.toFixed(2),
                    y: this.object.position.y.toFixed(2),
                    z: this.object.position.z.toFixed(2)
                },
                scale: this.object.scale.x.toFixed(2),
                type: this.currentModel ? 'GLB Model' : 'Default Icosahedron'
            });
        }
        
        this.object.visible = true;
    }

    lostCamera(){
        if(this.object.visible){
            console.log('📍 Tracking lost - model hidden');
        }
        this.object.visible = false;
    }
}

export class ARGroundProjectorView{
    constructor(container, width, height){
        this.applyPose = WebARCoreConnectorTHREE.Initialize(THREE);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setClearColor(0, 0);
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        this.camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 1000);
        this.raycaster = new THREE.Raycaster();

        this.ground = new THREE.Mesh(
            new THREE.CircleGeometry(1000, 64),
            new THREE.MeshBasicMaterial({
                color: 0xffffff,
                transparent: true,
                depthTest: true,
                opacity: 0.1,
                side: THREE.DoubleSide
            })
        );
        this.ground.rotation.x = Math.PI / 2;
        this.ground.position.y = -10;

        this.scene = new THREE.Scene();
        this.scene.add(new THREE.AmbientLight(0x808080));
        this.scene.add(new THREE.HemisphereLight(0x404040, 0xf0f0f0, 1));
        this.scene.add(this.ground);
        this.scene.add(this.camera);

        container.appendChild(this.renderer.domElement);

        const render = () => {
            requestAnimationFrame(render.bind(this));
            this.renderer.render(this.scene, this.camera);
        };

        render();
    }

    updateCameraPose(pose){
        this.applyPose(pose, this.camera.quaternion, this.camera.position);
        this.ground.position.x = this.camera.position.x;
        this.ground.position.z = this.camera.position.z;
        this.scene.children.forEach((obj) => obj.visible = true);
    }
    
    lostCamera(){
        this.scene.children.forEach((obj) => obj.visible = false);
    }

    addObjectAt(x, y, scale = 1.0){
        const el = this.renderer.domElement;
        const coord = new THREE.Vector2((x / el.offsetWidth) * 2 - 1, -(y / el.offsetHeight) * 2 + 1);
        this.raycaster.setFromCamera(coord, this.camera);

        const intersections = this.raycaster.intersectObjects([this.ground]);
        if(intersections.length > 0){
            const point = intersections[0].point;
            const object = new THREE.Mesh(
                new THREE.IcosahedronGeometry(1, 0),
                new THREE.MeshNormalMaterial({ flatShading: true })
            );
            object.scale.set(scale, scale, scale);
            object.position.set(point.x, point.y, point.z);
            object.custom = true;
            this.scene.add(object);
        }
    }

    reset(){
        // TODO: two loops????
        this.scene.children.filter((object) => object.custom).forEach((object) => this.scene.remove(object));
    }
}

export class ARTrackingRenderer{
    constructor(container, width, height, mapView = null){
        this.applyPose = WebARCoreConnectorTHREE.Initialize(THREE);
        
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setClearColor(0, 0);
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera.rotation.reorder('YXZ');
        this.camera.updateProjectionMatrix();

        this.scene = new THREE.Scene();
        this.scene.add(new THREE.AmbientLight(0x808080));
        this.scene.add(new THREE.HemisphereLight(0x404040, 0xf0f0f0, 1));
        this.scene.add(this.camera);

        this.body = document.body;
        container.appendChild(this.renderer.domElement);

        if(mapView){
            this.mapView = mapView;
            this.mapView.camHelper = new THREE.CameraHelper(this.camera);
            this.mapView.scene.add(this.mapView.camHelper);
        }
    }

    // TODO: use event listener for updateCameraPose && lostCamera
    updateCameraPose(pose){
        this.applyPose(pose, this.camera.quaternion, this.camera.position);
        this.renderer.render(this.scene, this.camera);
        this.body.classList.add('tracking');
    }

    lostCamera(){
        this.body.classList.remove('tracking');
    }

    createObjectWithPose(pose, scale = 1.0){
        const plane = new THREE.Mesh(
            new THREE.PlaneGeometry(scale, scale),
            new THREE.MeshBasicMaterial({
                color: 0xffffff,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.1
            })
        );

        scale *= 0.25;
        const cube = new THREE.Mesh(
            new THREE.BoxGeometry(scale, scale, scale),
            new THREE.MeshNormalMaterial({ flatShading: true })  
        );
        cube.position.z = scale * 0.5;
        plane.add(cube);
        plane.custom = true;
        this.applyPose(pose, plane.quaternion, plane.position);
        this.scene.add(plane);
        if(this.mapView){
            this.mapView.scene.add(plane.clone());
        }
    }

    reset(){
        this.scene.children.filter((object) => object.custom).forEach((object) => this.scene.remove(object));
        if(this.mapView){
            this.mapView.scene.children.filter((object) => object.custom).forEach((object) => this.mapView.scene.remove(object));
        }
    }
}

export class ARDebugMapView{
    constructor(container, width, height){
        this.renderer = new THREE.WebGLRenderer({ antialias: false });
        this.renderer.setClearColor(new THREE.Color('rgb(255, 255, 255)'));
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(width, height, false);
        this.renderer.domElement.style.width = width + 'px';
        this.renderer.domElement.style.height = height + 'px';
     
        this.camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 1000);
        this.camera.position.set(-1, 2, 2);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.1;
        this.controls.minDistance = 0.1;
        this.controls.maxDistance = 1000;

        this.gridHelper = new THREE.GridHelper(150, 100);
        this.gridHelper.position.y = -1;

        this.axisHelper = new THREE.AxesHelper(0.25);

        this.camHelper = null;

        this.scene = new THREE.Scene();
        this.scene.add(new THREE.AmbientLight(0xefefef));
        this.scene.add(new THREE.HemisphereLight(0x404040, 0xf0f0f0, 1));
        this.scene.add(this.gridHelper);
        this.scene.add(this.axisHelper);

        container.appendChild(this.renderer.domElement);

        const render = () => {
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
            requestAnimationFrame(render);
        };

        render();
    }
}