import express from 'express';
import https from 'https';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import selfsigned from 'selfsigned';
import mime from 'mime-types';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const rootPath = path.resolve(__dirname, '..');
const certsDir = path.join(__dirname, 'certs');
const keyPath = path.join(certsDir, 'key.pem');
const certPath = path.join(certsDir, 'cert.pem');

const app = express();

const ensureSelfSignedCert = () => {
    if(!fs.existsSync(keyPath) || !fs.existsSync(certPath)){
        console.log('Generating self-signed SSL certificate...');
        const attrs = [{ name: 'commonName', value: '0.0.0.0' }];
        const pems = selfsigned.generate(attrs, { days: 365 });
        fs.mkdirSync(certsDir, { recursive: true });
        fs.writeFileSync(certPath, pems.cert);
        fs.writeFileSync(pems.private);
    }
};

// Required headers for SharedArrayBuffer
app.use((req, res, next) => {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    next();
});

app.get('*', (req, res) => {
    const reqPath = decodeURIComponent(req.path);
    const fullPath = path.join(rootPath, reqPath);
    if(!fullPath.startsWith(rootPath)){
        return res.status(403).send('Access denied');
    }
    if(fs.existsSync(fullPath)){
        const stats = fs.statSync(fullPath);
        if(stats.isDirectory()){
            const files = fs.readdirSync(fullPath);
            const html = `
                <h1>Index of ${reqPath}</h1>
                <ul>
                    ${reqPath !== '/' ? `<li><a href="${path.join(reqPath, '..')}">..</a></li>` : ''}
                    ${files
                    .map(f => {
                        const href = path.join(reqPath, f).replace(/\\/g, '/');
                        const isDir = fs.statSync(path.join(fullPath, f)).isDirectory();
                        return `<li><a href="${href}${isDir ? '/' : ''}">${f}${isDir ? '/' : ''}</a></li>`;
                    })
                    .join('')}
                </ul>`;
            res.send(html);
        }else{
            const type = mime.lookup(fullPath) || 'application/octet-stream';
            res.type(type);
            fs.createReadStream(fullPath).pipe(res);
        }
    }else{
        res.status(404).send('Not Found');
    }
});

const sslOptions = {
    key: fs.readFileSync(keyPath),
    cert: fs.readFileSync(certPath)
};

ensureSelfSignedCert();

https.createServer(sslOptions, app).listen(8000, () => {
    console.log('🚀 HTTPS server at https://0.0.0.0:8000');
});