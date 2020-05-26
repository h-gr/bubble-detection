import tensorflow as tf
from tensorflow.keras.layers import Layer


class ROIPoolingLayer(Layer):
    """ Implements Region Of Interest Max Pooling 
        for channel-first images and relative bounding box coordinates
        
        # Constructor parameters
            pooled_height, pooled_width (int) -- 
              specify height and width of layer outputs
        
        Shape of inputs
            [(batch_size, pooled_height, pooled_width, n_channels),
             (batch_size, num_rois, 4)]
           
        Shape of output
            (batch_size, num_rois, pooled_height, pooled_width, n_channels)
    
    """
    def __init__(self, pooled_height, pooled_width, **kwargs):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        
        super(ROIPoolingLayer, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shape):
        """ Returns the shape of the ROI Layer output
        """
        feature_map_shape, rois_shape = input_shape
        assert feature_map_shape[0] == rois_shape[0]
        batch_size = feature_map_shape[0]
        n_rois = rois_shape[1]
        n_channels = feature_map_shape[3]
        return (batch_size, n_rois, self.pooled_height, 
                self.pooled_width, n_channels)

    def call(self, x):
        """ Maps the input tensor of the ROI layer to its output
        
            # Parameters
                x[0] -- Convolutional feature map tensor,
                        shape (batch_size, pooled_height, pooled_width, n_channels)
                x[1] -- Tensor of region of interests from candidate bounding boxes,
                        shape (batch_size, num_rois, 4)
                        Each region of interest is defined by four relative 
                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1
            # Output
                pooled_areas -- Tensor with the pooled region of interest, shape
                    (batch_size, num_rois, pooled_height, pooled_width, n_channels)
        """
        def curried_pool_rois(x): 
          return ROIPoolingLayer._pool_rois(x[0], x[1], 
                                            self.pooled_height, 
                                            self.pooled_width)
        
        pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)

        return pooled_areas
    
    @staticmethod
    def _pool_rois(feature_map, rois, pooled_height, pooled_width):
        """ Applies ROI pooling for a single image and varios ROIs
        """
        def curried_pool_roi(roi): 
          return ROIPoolingLayer._pool_roi(feature_map, roi, 
                                           pooled_height, pooled_width)
        
        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
        return pooled_areas
    
    @staticmethod
    def _pool_roi(feature_map, roi, pooled_height, pooled_width):
        """ Applies ROI pooling to a single image and a single region of interest
        """

        # Compute the region of interest        
        feature_map_height = int(feature_map.shape[0])
        feature_map_width  = int(feature_map.shape[1])
        
        h_start = tf.cast(feature_map_height * roi[0], 'int32')
        w_start = tf.cast(feature_map_width  * roi[1], 'int32')
        h_end   = tf.cast(feature_map_height * roi[2], 'int32')
        w_end   = tf.cast(feature_map_width  * roi[3], 'int32')
        

        # Divide the region into non overlapping areas
        region_height = h_end - h_start
        region_width  = w_end - w_start
        
        region = feature_map[h_start:h_end, w_start:w_end, :]

        def dynamic_padding(inp, min_size,dimen):

          pad_size = min_size - tf.shape(inp)[dimen]
          paddings=[]
          if dimen == 0:
            paddings = [[(pad_size/2)+1,(pad_size/2)+1], [0, 0], [0, 0]] # assign here, during graph execution
          else:
            paddings = [[0, 0], [(pad_size/2)+1,(pad_size/2)+1], [0, 0]] # assign here, during graph execution
          return tf.pad(inp, paddings,'CONSTANT', constant_values=255)

        # Pad only if necessary
        region = tf.cond(tf.less(tf.shape(region)[1], 10), true_fn=lambda: dynamic_padding(region, 10,1), false_fn=lambda: region) 
        region = tf.cond(tf.less(tf.shape(region)[0], 10), true_fn=lambda: dynamic_padding(region, 10,0), false_fn=lambda: region) 

        region_height = tf.shape(region)[0]
        region_width =  tf.shape(region)[1]

        h_step = tf.cast( region_height / pooled_height, 'int32')
        w_step = tf.cast( region_width  / pooled_width , 'int32')
        
        areas = [[(
                    i*h_step, 
                    j*w_step, 
                    (i+1)*h_step if i+1 < pooled_height else region_height, 
                    (j+1)*w_step if j+1 < pooled_width else region_width
                   ) 
                   for j in range(pooled_width)] 
                  for i in range(pooled_height)]
        
        # take the maximum of each area and stack the result
        def pool_area(x): 
          return tf.math.reduce_min(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])
        
        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
        return pooled_features
        
import json
def loadjson(path):
    with open(path, 'r') as f:
        distros_dict = json.load(f)
    return distros_dict


import numpy as np# Define parameters
from PIL import Image
import tensorflow as tf
import os

#tf.disable_v2_behavior()
def obtainInOut (pathImg, pathJson):
  Xout = []
  Xcoordout = []
  Yout = []
  arr = os.listdir(pathImg)
  for fi in arr:
      if(fi.__contains__(".png")):
        inputImg=Image.open(pathImg + "/" + fi)
        dic = loadjson(pathJson)
        reg = dic[fi]
        #converting rois
        rect = []
        circ = []
        for r in reg:
          rtgd = r['rectangular']
          xmin = rtgd['x']
          xmax = rtgd['x']+rtgd['width']
          ymin = rtgd['y']
          ymax = rtgd['y']+rtgd['height']
          coord = np.asarray([ymin,xmin,ymax,xmax], dtype='float32')
          rect.append(coord/128)

          crcd = r['circular']
          xc=crcd['cx']
          yc=crcd['cy']
          r=crcd['r']
          coordc = np.asarray([xc,yc,r], dtype='float32')
          circ.append(coordc)

        yresult = np.asarray(circ,dtype='float32')
        Yout.append(yresult)

        xresult = np.asarray(rect,dtype='float32')*128
        Xcoordout.append(xresult)

        rectarr = np.asarray([rect],dtype='float32')

        #print(rectarr)

        batch_size = 1
        img_height = 128
        img_width = 128
        n_channels = 3
        n_rois = len(reg)
        pooled_height = 10
        pooled_width = 10# Create feature map input
        feature_maps_shape = (batch_size, img_height, img_width, n_channels)
        
        feature_maps_np = np.asarray([np.asarray(inputImg)],dtype='float32')
        #print(f"feature_maps_np.shape = {feature_maps_np.shape}")# Create batch size
        
        roiss_np = rectarr
        #print(f"roiss_np.shape = {roiss_np.shape}")# Create layer
        roi_layer = ROIPoolingLayer(pooled_height, pooled_width)
        #print(f"output shape of layer call = {pooled_features.shape}")# Run tensorflow session
        result = roi_layer([feature_maps_np, roiss_np])
        Xout.append(result[0,:,:,:,:])

        # print(f"result.shape = {result.shape}")
        # for i in range(24):
        #   print(i)
        #   print(f"first  roi embedding=\n{np.mean(result[0,i,:,:,:], axis=2)}")
        #   img = np.mean(result[0,i,:,:,:], axis=2)
        #   print(img.shape)
        #   img = Image.fromarray(np.uint8(img), 'L')
        #   img = img.resize((200,200))
        #   img.save('{}.png'.format(i))

        #   xx = rectarr[0][i]*128
        #   print(xx[2]-xx[0])
        #   print(xx[3]-xx[1])
        #   im1 = inputImg.crop((xx[1], xx[0], xx[3], xx[2])) 
        #   im1.save('crop{}.png'.format(i))
  Xout = np.concatenate( Xout, axis=0 )
  Yout = np.concatenate( Yout, axis=0 )
  Xcoordout = np.concatenate( Xcoordout, axis=0 )
  return Xout, Yout, Xcoordout