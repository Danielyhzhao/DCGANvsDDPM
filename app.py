from flask import Flask, request, send_file, Response
import torch
from model import Generator, ConditionalUNet, DiffusionModel
from PIL import Image
import io
import time
import os

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = 'C:/Users/Daniel ZHAO/AIBA/foundation_ai/Essay'

# 加载最佳预训练模型
try:
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(os.path.join(save_dir, 'dcgan_generator_best.pth'), map_location=device))
    generator.eval()
    unet = ConditionalUNet(num_classes=10).to(device)
    unet.load_state_dict(torch.load(os.path.join(save_dir, 'ddpm_unet_best.pth'), map_location=device))
    unet.eval()
    diffusion = DiffusionModel(unet, device, T=100)  # 默认 T，可根据训练结果调整
except FileNotFoundError:
    raise Exception("未找到模型文件。请先运行 train.py 生成模型文件。")

@app.route('/')
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Conditional DCGAN and DDPM Image Generator</title>
        <script src="https://cdn.jsdelivr.net/npm/react@17/umd/react.development.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/react-dom@17/umd/react-dom.development.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.22.5/babel.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body class="bg-gradient-to-r from-blue-100 to-gray-100 min-h-screen flex items-center justify-center">
        <div id="root"></div>
        <script type="text/babel">
            function ImageGenerator() {
                const [model, setModel] = React.useState('dcgan');
                const [classIdx, setClassIdx] = React.useState('0');
                const [imageSrc, setImageSrc] = React.useState('');
                const [inferenceTime, setInferenceTime] = React.useState('');
                const [isLoading, setIsLoading] = React.useState(false);
                const [error, setError] = React.useState('');

                const handleGenerate = () => {
                    setIsLoading(true);
                    setError('');
                    setImageSrc('');
                    fetch('/generate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ model: model, class_idx: parseInt(classIdx) })
                    })
                    .then(response => {
                        if (!response.ok) throw new Error('生成图像失败，请重试。');
                        const contentDisp = response.headers.get('Content-Disposition');
                        const filenameMatch = contentDisp?.match(/filename="(.+?)"/);
                        const filename = filenameMatch ? filenameMatch[1] : '';
                        const timeMatch = filename.match(/inference_(\d+\.\d+)s/);
                        const time = timeMatch ? timeMatch[1] : '未知';
                        setInferenceTime(time);
                        return response.blob();
                    })
                    .then(blob => {
                        const url = URL.createObjectURL(blob);
                        setImageSrc(url);
                        setIsLoading(false);
                    })
                    .catch(error => {
                        setError(error.message);
                        setIsLoading(false);
                    });
                };

                return (
                    <div className="bg-white p-8 rounded-lg shadow-xl w-full max-w-md">
                        <h1 className="text-3xl font-bold mb-4 text-center text-blue-600">图像生成器</h1>
                        <p className="text-gray-600 text-center mb-6">使用最佳 DCGAN 或 DDPM 模型生成手写数字</p>
                        <div className="mb-4">
                            <label htmlFor="model" className="block text-sm font-medium text-gray-700">选择模型:</label>
                            <select 
                                id="model" 
                                value={model} 
                                onChange={(e) => setModel(e.target.value)} 
                                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                            >
                                <option value="dcgan">最佳 DCGAN</option>
                                <option value="ddpm">最佳 DDPM</option>
                            </select>
                        </div>
                        <div className="mb-6">
                            <label htmlFor="class" className="block text-sm font-medium text-gray-700">选择数字:</label>
                            <select 
                                id="class" 
                                value={classIdx} 
                                onChange={(e) => setClassIdx(e.target.value)} 
                                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                            >
                                {[...Array(10).keys()].map(i => (
                                    <option key={i} value={i}>{i}</option>
                                ))}
                            </select>
                        </div>
                        <button 
                            onClick={handleGenerate} 
                            className="w-full bg-blue-600 hover:bg-blue-800 text-white font-semibold py-2 px-4 rounded-lg transition duration-300"
                            disabled={isLoading}
                        >
                            {isLoading ? '生成中...' : '生成图像'}
                        </button>
                        {isLoading && (
                            <div className="mt-6 flex justify-center">
                                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-600"></div>
                            </div>
                        )}
                        {error && (
                            <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-md text-center">
                                {error}
                            </div>
                        )}
                        {imageSrc && !isLoading && (
                            <div className="mt-6">
                                <img src={imageSrc} alt="Generated Image" className="w-full rounded-lg shadow-md" />
                                <p className="text-center mt-2 text-gray-600">推理时间: {inferenceTime} 秒</p>
                            </div>
                        )}
                    </div>
                );
            }

            ReactDOM.render(<ImageGenerator />, document.getElementById('root'));
        </script>
    </body>
    </html>
    """
    return Response(html_content, mimetype='text/html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    model_type = data.get('model')
    class_idx = int(data.get('class_idx', 0))

    start_time = time.time()
    if model_type == 'dcgan':
        z = torch.randn(1, 100).to(device)
        labels = torch.tensor([class_idx]).to(device)
        with torch.no_grad():
            img = generator(z, labels).cpu()
    elif model_type == 'ddpm':
        labels = torch.tensor([class_idx]).to(device)
        with torch.no_grad():
            img = diffusion.sample(1, labels).cpu()
    else:
        return "无效的模型类型", 400
    inference_time = time.time() - start_time

    img = (img + 1) / 2 * 255
    img = img.squeeze().numpy().astype('uint8')
    img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png', as_attachment=False,
                     attachment_filename=f'{model_type}_inference_{inference_time:.4f}s.png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)