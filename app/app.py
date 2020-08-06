import torch
import torchvision.transforms as transforms
from model import CNNtoRNN
from PIL import Image
import pickle
import os
import io
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import requests
import os.path
from os import path

f = open("vocab_itos.pkl", "rb")
vocab = pickle.load(f)

embed_size = 256
hidden_size = 256
vocab_size = len(vocab)
num_layers = 1
MODEL_PATH = os.getenv('MODEL_PATH')  #"my_checkpoint.pth.tar"
MODEL_URL =  os.getenv('MODEL_URL') #"https://vonage-models.s3.amazonaws.com/my_checkpoint.pth.tar"

if not path.exists(MODEL_PATH):
    print("downloading model....")
    r = requests.get(MODEL_URL)
    open(MODEL_PATH, 'wb').write(r.content)

print('done!\nloading up the saved model weights...')

myModel = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to("cpu")
myModel.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))['state_dict'])
myModel.eval()

app = Flask(__name__)

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def transform_image(img_path):
    my_transforms = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = Image.open(img_path)
    return my_transforms(image).unsqueeze(0)
    
def get_caption(img_path):
    image = transform_image(img_path)
    caption = " ".join(myModel.caption_image(image.to("cpu"), vocab))
    return caption

@app.route('/', methods=['GET', 'POST'])
def index():
   if request.method == 'POST':
       if 'file' not in request.files:
           print('No file attached in request')
           return redirect(request.url)
       file = request.files['file']
       if file.filename == '':
           print('No file selected')
           return redirect(request.url)
       if file and allowed_file(file.filename):
           filename = secure_filename(file.filename)
           path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
           print("file uploaded ",path)
           file.save(path)
           caption = get_caption(img_path=path)
           return render_template('caption.html', image_url="uploads/"+filename, caption=caption)
                
   return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
