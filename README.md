# bubble-detection

This approach uses a combination of RPN and a neural network using mixed data to predict circle parameters for foam bubble images  

Predices bubbles (position, radius) within an image)

data : Images and annotations. In general the images are annotated with VGG Image Annotation

mmdetection: API to train different kinds of RPN. In this case I use Faster RCNN.

models: best trained models.

scripts: useful functions and special architecture definitions.

mmdet needs to be installed first. so if it is already installed in you computer, the installation lines in the notebooks can be commented.

Training.ipynb: Notebook that shows how the system is trained.

  google colab : https://colab.research.google.com/drive/10dlwMwB8OMHbGxk9sX7qcR9EYfyDQI0U?usp=sharing
  
  
Inference.ipynb: Notebook to use a trained system with new images.

  google colab : https://colab.research.google.com/drive/1oNfikDvby-7nl-YyKM5B6WjhrXWKAHrF?usp=sharing
  
