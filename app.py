# app.py
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

# Load pre-trained models
try:
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(os.path.join(save_dir, 'dcgan_generator.pth'), map_location=device))
    generator.eval()
    unet = ConditionalUNet(num_classes=10).to(device)
    unet.load_state_dict(torch.load(os.path.join(save_dir, 'ddpm_unet.pth'), map_location=device))
    unet.eval()
    diffusion = DiffusionModel(unet, device)
except FileNotFoundError:
    raise Exception("Model files not found. Please run train.py to generate the model files.")

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
                        if (!response.ok) throw new Error('Failed to generate image. Please try again.');
                        const contentDisp = response.headers.get('Content-Disposition');
                        const filenameMatch = contentDisp?.match(/filename="(.+?)"/);
                        const filename = filenameMatch ? filenameMatch[1] : '';
                        const timeMatch = filename.match(/inference_(\d+\.\d+)s/);
                        const time = timeMatch ? timeMatch[1] : 'unknown';
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
                        <h1 className="text-3xl font-bold mb-4 text-center text-blue-600">Image Generator</h1>
                        <p className="text-gray-600 text-center mb-6">Generate handwritten digits using DCGAN or DDPM models</p>
                        <div className="mb-4">
                            <label htmlFor="model" className="block text-sm font-medium text-gray-700">Select Model:</label>
                            <select 
                                id="model" 
                                value={model} 
                                onChange={(e) => setModel(e.target.value)} 
                                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                            >
                                <option value="dcgan">DCGAN</option>
                                <option value="ddpm">DDPM</option>
                            </select>
                        </div>
                        <div className="mb-6">
                            <label htmlFor="class" className="block text-sm font-medium text-gray-700">Select Digit:</label>
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
                            {isLoading ? 'Generating...' : 'Generate Image'}
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
                                <p className="text-center mt-2 text-gray-600">Inference Time: {inferenceTime} seconds</p>
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
        return "Invalid model type", 400
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