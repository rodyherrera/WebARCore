class WebGL2Utils{
    static compileShader(glContext, sourceCode, shaderType){
        const shader = glContext.createShader(shaderType);
        glContext.shaderSource(shader, sourceCode);
        glContext.compileShader(shader);

        if(!glContext.getShaderParameter(shader, glContext.COMPILE_STATUS)){
            const errorMessage = glContext.getShaderInfoLog(shader);
            const formattedSource = sourceCode.split('\n').map((line, index) => `${index + 1}: ${line}`).join('\n');
            glContext.deleteShader(shader);
            
            throw new Error(`Shader compilation failed: ${errorMessage}\n${formattedSource}`);
        }

        return shader;
    }

    static createShaderProgram(glContext, vertexShaderCode, fragmentShaderCode){
        const vertexShader = WebGL2Utils.compileShader(glContext, vertexShaderCode, glContext.VERTEX_SHADER);
        const fragmentShader = WebGL2Utils.compileShader(glContext, fragmentShaderCode, glContext.FRAGMENT_SHADER);

        const shaderProgram = glContext.createProgram();
        glContext.attachShader(shaderProgram, fragmentShader);
        glContext.attachShader(shaderProgram, vertexShader);

        glContext.linkProgram(shaderProgram);
        glContext.validateProgram(shaderProgram);

        if(!glContext.getProgramParameter(shaderProgram, glContext.LINK_STATUS)){
            glContext.deleteProgram(shaderProgram);
            glContext.deleteShader(fragmentShader);
            glContext.deleteShader(vertexShader);
            throw new Error('Shader program linking failed: ' + glContext.getProgramInfoLog(shaderProgram));
        }

        return shaderProgram;
    }

    static createTexture2D(glContext, width, height, pixelType, flipY = false, useNearestFilter = false){
        const texture = glContext.createTexture();
        glContext.bindTexture(glContext.TEXTURE_2D, texture);
        glContext.pixelStorei(glContext.UNPACK_FLIP_Y_WEBGL, flipY);

        glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_WRAP_S, glContext.CLAMP_TO_EDGE);
        glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_WRAP_T, glContext.CLAMP_TO_EDGE);
        glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_MIN_FILTER, useNearestFilter ? glContext.NEAREST : glContext.LINEAR);
        glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_MAG_FILTER, useNearestFilter ? glContext.NEAREST : glContext.LINEAR);

        const internalFormat = pixelType === glContext.FLOAT ? glContext.RGBA32F : glContext.RGBA;
        glContext.texImage2D(glContext.TEXTURE_2D, 0, internalFormat, width, height, 0, glContext.RGBA, pixelType || glContext.UNSIGNED_BYTE, null);
        return texture;
    }

    static createFramebufferWithTexture(glContext, width, height, pixelType = undefined, flipY = false, useNearestFilter = false){
        const texture = WebGL2Utils.createTexture2D(glContext, width, height, pixelType, flipY, useNearestFilter);
        const framebuffer = glContext.createFramebuffer();

        glContext.bindFramebuffer(glContext.FRAMEBUFFER, framebuffer);
        glContext.framebufferTexture2D(glContext.FRAMEBUFFER, glContext.COLOR_ATTACHMENT0, glContext.TEXTURE_2D, texture, 0);

        if(glContext.checkFramebufferStatus(glContext.FRAMEBUFFER) !== glContext.FRAMEBUFFER_COMPLETE){
            throw new Error('Framebuffer setup failed.');
        }

        return {
            texture: texture,
            framebuffer: framebuffer
        };
    }

    static getWebGLContextInfo(glContext){
        const info = {
            renderer: glContext.getParameter(glContext.RENDERER),
            vendor: glContext.getParameter(glContext.VENDOR),
            version: glContext.getParameter(glContext.VERSION),
            shadingLanguageVersion: glContext.getParameter(glContext.SHADING_LANGUAGE_VERSION),
            unmaskedRenderer: '-',
            unmaskedVendor: '-'
        };

        const debugInfo = glContext.getExtension('WEBGL_debug_renderer_info');
        if(debugInfo){
            info.unmaskedRenderer = glContext.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
            info.unmaskedVendor = glContext.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
        }

        return info;
    }


    static async fetchShaderSource(url){
        const response = await fetch(url);
        return await response.text();
    }
}

export { WebGL2Utils };