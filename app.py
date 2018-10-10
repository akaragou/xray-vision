import os
from scipy.misc import imsave
from flask import Flask, render_template, flash, request, redirect, url_for, jsonify, send_from_directory, make_response
from werkzeug.utils import secure_filename
from functools import wraps, update_wrapper
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import base64
import cv2
import StringIO
import matplotlib
from data_utils import centered_crop, maintain_aspec_ratio_resize
from datetime import datetime

UPLOAD_FOLDER = 'uploads/'
PROCESS_FOLDER = 'process_results/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESS_FOLDER'] = PROCESS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

def maintain_aspec_ratio_resize(img, desired_size=256):
  old_size = img.size  
  ratio = float(desired_size)/max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])

  img = img.resize(new_size, Image.ANTIALIAS)

  new_img = Image.new("RGB", (desired_size, desired_size))
  new_img.paste(img, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))

  new_img = np.array(new_img)
  return new_img

def nocache(view):
  @wraps(view)
  def no_cache(*args, **kwargs):
    response = make_response(view(*args, **kwargs))
    response.headers['Last-Modified'] = datetime.now()
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
        
  return update_wrapper(no_cache, view)

def allowed_file(filename):
  return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
@nocache
def upload_file():
  
  if request.method == 'POST':
    # check if the post request has the file part
    if 'file' not in request.files:
      # flash('No file part')
      return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
      # flash('No selected file')
      return redirect(request.url)
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      # return redirect(url_for('upload_file',
      #                         filename=filename))
      return process_uploaded_image(filename)

  return render_template('index.html')

def load_graph(trained_model):   
  with tf.gfile.GFile(trained_model, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        graph_def,
        input_map=None,
        return_elements=None,
        name=""
        )
  return graph

@app.route('/process_results/<filename>')
@nocache
def send_file(filename):
    return send_from_directory(PROCESS_FOLDER, filename)

@nocache
def process_uploaded_image(filename):
  img = Image.open(os.path.join(UPLOAD_FOLDER, filename))
  img = maintain_aspec_ratio_resize(img)
  img = centered_crop(img, 224, 224)
  img = img[np.newaxis,:,:,:]
  img = img/255.0

  label = [1]
  graph = app.graph
  ## NOW the complete graph with values has been restored
  inputs = graph.get_tensor_by_name("inputs:0")
  labels = graph.get_tensor_by_name("labels:0")
  preprocessed_inputs = graph.get_tensor_by_name("densenet121/imagenet_preprocessing/image_bgr:0")

  masked_logits = graph.get_tensor_by_name("densenet121/masked_logits:0")
  prob = graph.get_tensor_by_name("densenet121/probability:0")

  gradient = tf.gradients(masked_logits, preprocessed_inputs, name='gradients')

  y_test_images = np.zeros((1, 2))
  sess = tf.Session(graph=graph)
 
  grad_im_list = []
  for _ in range(64):
    eps = np.random.normal(0.0, 0.0125, [224, 224, 3])
    
    image_eps = img + eps
    
    grad_im_list += [sess.run(gradient,feed_dict={inputs: image_eps,
                                                    labels: label})]

  M = np.mean(np.concatenate(grad_im_list), axis=0) # M is the smooth grad image, checkout https://arxiv.org/abs/1706.03825 
  # normalizing by taking the absolute value per pixel 
  # and the suming across all three chanels 
  abs_M = np.abs(M)
  sum_M = np.sum(abs_M, axis=3)
  # prediction for current batch image
  mask = np.squeeze(sum_M)
  thres = np.percentile(mask.ravel(), 99)
  idx = mask[:,:] < thres
  mask[idx] = 0
  kernel = np.ones((5,5),np.float32)/25
  mask = cv2.filter2D(mask,-1,kernel)
  to_plot_img = np.squeeze(img)
  to_plot_img = cv2.resize(to_plot_img, (600, 600), cv2.INTER_LINEAR)
  mask = cv2.resize(mask, (600, 600), cv2.INTER_LINEAR)
  plt.figure()
  plt.imshow(to_plot_img, interpolation='none')
  plt.imshow(mask, interpolation='none', alpha=0.35)
  plt.gca().set_axis_off()
  plt.gca().xaxis.set_major_locator(matplotlib.ticker.NullLocator())
  plt.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())
  plt.savefig('process_results/sensitivity_map.png', bbox_inches='tight', pad_inches=0)
  plt.close()
  imsave('process_results/input_img.png', to_plot_img)
  predictions = sess.run(prob, feed_dict={inputs: img})
  
  out = {"normal":str(round(predictions[0][0]*100,1)),"abnormal":str(round(predictions[0][1]*100,1))}
  sess.close()
  return render_template('process.html', prob=out["abnormal"], input_img='input_img.png', sensitivity_map='sensitivity_map.png')

app.graph = load_graph('./frozen_graph.pb')  
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("5000"))
