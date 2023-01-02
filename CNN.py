#!/usr/bin/env python
# coding: utf-8

# In[8]:


from keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator,load_img
from models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'cell_images/Train'
valid_path = 'cell_images/Test'

mobilnet = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in mobilnet.layers:
    layer.trainable = False
  
folders = glob('Dataset/Train/*')
folders
['Dataset/Train\\Parasite', 'Dataset/Train\\Uninfected']

x = Flatten()(mobilnet.output)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=mobilnet.input, outputs=prediction)
# view the structure of the model
model.summary()
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 25088)             0         
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 50178     
=================================================================
Total params: 20,074,562
Trainable params: 50,178
Non-trainable params: 20,024,384
_________________________________________________________________
from tensorflow.keras.layers import MaxPooling2D
### Create Model from scratch using CNN
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.summary()
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 224, 224, 16)      208       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 112, 112, 16)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 112, 112, 32)      2080      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 56, 56, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 56, 56, 64)        8256      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 28, 28, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 50176)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 500)               25088500  
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 1002      
=================================================================
Total params: 25,100,046
Trainable params: 25,100,046
Non-trainable params: 0
_________________________________________________________________
# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('Dataset/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
Found 416 images belonging to 2 classes.
training_set
test_set = test_datagen.flow_from_directory('Dataset/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
Found 134 images belonging to 2 classes.

r = model..fit_generator(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
WARNING:tensorflow:sample_weight modes were coerced from
  ...
    to  
  ['...']
WARNING:tensorflow:sample_weight modes were coerced from
  ...
    to  
  ['...']
Train for 13 steps, validate for 5 steps
Epoch 1/50
13/13 [==============================] - 6s 490ms/step - loss: 1.4529 - accuracy: 0.5048 - val_loss: 0.9426 - val_accuracy: 0.4179
Epoch 2/50
13/13 [==============================] - 6s 498ms/step - loss: 0.7403 - accuracy: 0.6010 - val_loss: 0.6418 - val_accuracy: 0.6791
Epoch 3/50
13/13 [==============================] - 6s 495ms/step - loss: 0.5256 - accuracy: 0.7524 - val_loss: 0.9822 - val_accuracy: 0.5224
Epoch 4/50
13/13 [==============================] - 6s 459ms/step - loss: 0.4426 - accuracy: 0.7788 - val_loss: 0.4205 - val_accuracy: 0.7761
Epoch 5/50
13/13 [==============================] - 6s 466ms/step - loss: 0.3296 - accuracy: 0.8630 - val_loss: 0.5563 - val_accuracy: 0.7015
Epoch 6/50
13/13 [==============================] - 6s 473ms/step - loss: 0.3186 - accuracy: 0.8654 - val_loss: 0.3651 - val_accuracy: 0.8358
Epoch 7/50
13/13 [==============================] - 6s 468ms/step - loss: 0.2774 - accuracy: 0.9014 - val_loss: 0.3622 - val_accuracy: 0.7910
Epoch 8/50
13/13 [==============================] - 6s 473ms/step - loss: 0.2810 - accuracy: 0.8822 - val_loss: 0.3307 - val_accuracy: 0.8433
Epoch 9/50
13/13 [==============================] - 6s 475ms/step - loss: 0.2592 - accuracy: 0.9135 - val_loss: 0.3152 - val_accuracy: 0.8284
Epoch 10/50
13/13 [==============================] - 6s 463ms/step - loss: 0.2235 - accuracy: 0.9279 - val_loss: 0.2885 - val_accuracy: 0.8881
Epoch 11/50
13/13 [==============================] - 6s 480ms/step - loss: 0.2190 - accuracy: 0.9255 - val_loss: 0.2744 - val_accuracy: 0.8731
Epoch 12/50
13/13 [==============================] - 6s 464ms/step - loss: 0.2206 - accuracy: 0.9303 - val_loss: 0.3062 - val_accuracy: 0.8731
Epoch 13/50
13/13 [==============================] - 6s 474ms/step - loss: 0.1917 - accuracy: 0.9495 - val_loss: 0.2626 - val_accuracy: 0.8881
Epoch 14/50
13/13 [==============================] - 6s 484ms/step - loss: 0.2047 - accuracy: 0.9351 - val_loss: 0.3098 - val_accuracy: 0.8507
Epoch 15/50
13/13 [==============================] - 6s 490ms/step - loss: 0.2383 - accuracy: 0.8966 - val_loss: 0.2602 - val_accuracy: 0.9104
Epoch 16/50
13/13 [==============================] - 6s 481ms/step - loss: 0.2286 - accuracy: 0.9087 - val_loss: 0.3670 - val_accuracy: 0.7910
Epoch 17/50
13/13 [==============================] - 6s 478ms/step - loss: 0.1809 - accuracy: 0.9423 - val_loss: 0.2570 - val_accuracy: 0.8955
Epoch 18/50
 2/13 [===>..........................] - ETA: 5s - loss: 0.1943 - accuracy: 0.9062
---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
 in 
      6   epochs=50,
      7   steps_per_epoch=len(training_set),
----> 8   validation_steps=len(test_set)
      9 )

~\Anaconda3\lib\site-packages\tensorflow_core\python\util\deprecation.py in new_func(*args, **kwargs)
    322               'in a future version' if date is None else ('after %s' % date),
    323               instructions)
--> 324       return func(*args, **kwargs)
    325     return tf_decorator.make_decorator(
    326         func, new_func, 'deprecated',

~\Anaconda3\lib\site-packages\tensorflow_core\python\keras\engine\training.py in fit_generator(self, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, validation_freq, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)
   1304         use_multiprocessing=use_multiprocessing,
   1305         shuffle=shuffle,
-> 1306         initial_epoch=initial_epoch)
   1307 
   1308   @deprecation.deprecated(

~\Anaconda3\lib\site-packages\tensorflow_core\python\keras\engine\training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
    817         max_queue_size=max_queue_size,
    818         workers=workers,
--> 819         use_multiprocessing=use_multiprocessing)
    820 
    821   def evaluate(self,

~\Anaconda3\lib\site-packages\tensorflow_core\python\keras\engine\training_v2.py in fit(self, model, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
    340                 mode=ModeKeys.TRAIN,
    341                 training_context=training_context,
--> 342                 total_epochs=epochs)
    343             cbks.make_logs(model, epoch_logs, training_result, ModeKeys.TRAIN)
    344 

~\Anaconda3\lib\site-packages\tensorflow_core\python\keras\engine\training_v2.py in run_one_epoch(model, iterator, execution_function, dataset_size, batch_size, strategy, steps_per_epoch, num_samples, mode, training_context, total_epochs)
    126         step=step, mode=mode, size=current_batch_size) as batch_logs:
    127       try:
--> 128         batch_outs = execution_function(iterator)
    129       except (StopIteration, errors.OutOfRangeError):
    130         # TODO(kaftan): File bug about tf function and errors.OutOfRangeError?

~\Anaconda3\lib\site-packages\tensorflow_core\python\keras\engine\training_v2_utils.py in execution_function(input_fn)
     96     # `numpy` translates Tensors to values in Eager mode.
     97     return nest.map_structure(_non_none_constant_value,
---> 98                               distributed_function(input_fn))
     99 
    100   return execution_function

~\Anaconda3\lib\site-packages\tensorflow_core\python\eager\def_function.py in __call__(self, *args, **kwds)
    566         xla_context.Exit()
    567     else:
--> 568       result = self._call(*args, **kwds)
    569 
    570     if tracing_count == self._get_tracing_count():

~\Anaconda3\lib\site-packages\tensorflow_core\python\eager\def_function.py in _call(self, *args, **kwds)
    597       # In this case we have created variables on the first call, so we run the
    598       # defunned version which is guaranteed to never create variables.
--> 599       return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
    600     elif self._stateful_fn is not None:
    601       # Release the lock early so that multiple threads can perform the call

~\Anaconda3\lib\site-packages\tensorflow_core\python\eager\function.py in __call__(self, *args, **kwargs)
   2361     with self._lock:
   2362       graph_function, args, kwargs = self._maybe_define_function(args, kwargs)
-> 2363     return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access
   2364 
   2365   @property

~\Anaconda3\lib\site-packages\tensorflow_core\python\eager\function.py in _filtered_call(self, args, kwargs)
   1609          if isinstance(t, (ops.Tensor,
   1610                            resource_variable_ops.BaseResourceVariable))),
-> 1611         self.captured_inputs)
   1612 
   1613   def _call_flat(self, args, captured_inputs, cancellation_manager=None):

~\Anaconda3\lib\site-packages\tensorflow_core\python\eager\function.py in _call_flat(self, args, captured_inputs, cancellation_manager)
   1690       # No tape is watching; skip to running the function.
   1691       return self._build_call_outputs(self._inference_function.call(
-> 1692           ctx, args, cancellation_manager=cancellation_manager))
   1693     forward_backward = self._select_forward_and_backward_functions(
   1694         args,

~\Anaconda3\lib\site-packages\tensorflow_core\python\eager\function.py in call(self, ctx, args, cancellation_manager)
    543               inputs=args,
    544               attrs=("executor_type", executor_type, "config_proto", config),
--> 545               ctx=ctx)
    546         else:
    547           outputs = execute.execute_with_cancellation(

~\Anaconda3\lib\site-packages\tensorflow_core\python\eager\execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     59     tensors = pywrap_tensorflow.TFE_Py_Execute(ctx._handle, device_name,
     60                                                op_name, inputs, attrs,
---> 61                                                num_outputs)
     62   except core._NotOkStatusException as e:
     63     if name is not None:

KeyboardInterrupt: 
 
# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('model_vgg19.h5')
 
y_pred = model.predict(test_set)
y_pred
array([[9.99876857e-01, 1.23175341e-04],
       [9.99977112e-01, 2.29190919e-05],
       [7.58346558e-01, 2.41653457e-01],
       [9.99925494e-01, 7.45595753e-05],
       [9.90784764e-01, 9.21520963e-03],
       [9.12076458e-02, 9.08792377e-01],
       [2.70295113e-01, 7.29704857e-01],
       [3.21944878e-02, 9.67805505e-01],
       [9.62613881e-01, 3.73861678e-02],
       [5.13265312e-01, 4.86734688e-01],
       [9.87143576e-01, 1.28563549e-02],
       [9.97780263e-01, 2.21971911e-03],
       [9.32238042e-01, 6.77619576e-02],
       [9.21115577e-01, 7.88843632e-02],
       [3.17853913e-02, 9.68214571e-01],
       [1.00000000e+00, 9.47071488e-09],
       [2.65438944e-01, 7.34561086e-01],
       [9.99419808e-01, 5.80202264e-04],
       [9.40651655e-01, 5.93483634e-02],
       [9.85742450e-01, 1.42575456e-02],
       [9.99954581e-01, 4.54339461e-05],
       [9.99656916e-01, 3.43117106e-04],
       [5.88945560e-02, 9.41105425e-01],
       [9.99902725e-01, 9.72729686e-05],
       [7.31766939e-01, 2.68233061e-01],
       [9.99747813e-01, 2.52151629e-04],
       [1.64933771e-01, 8.35066259e-01],
       [9.99608934e-01, 3.91100359e-04],
       [9.99930501e-01, 6.94626651e-05],
       [9.82600868e-01, 1.73991676e-02],
       [9.97824430e-01, 2.17563682e-03],
       [8.57747495e-02, 9.14225221e-01],
       [8.22690725e-01, 1.77309304e-01],
       [2.20711201e-01, 7.79288828e-01],
       [1.74620345e-01, 8.25379610e-01],
       [9.97595489e-01, 2.40448676e-03],
       [9.78304744e-01, 2.16952953e-02],
       [9.96270061e-01, 3.72986798e-03],
       [9.99968529e-01, 3.14996905e-05],
       [2.06163585e-01, 7.93836474e-01],
       [2.74875015e-01, 7.25125015e-01],
       [6.29736423e-01, 3.70263547e-01],
       [7.74357736e-01, 2.25642264e-01],
       [9.99867320e-01, 1.32690737e-04],
       [9.60264862e-01, 3.97351012e-02],
       [7.71383643e-01, 2.28616387e-01],
       [6.15290999e-01, 3.84708971e-01],
       [9.99967098e-01, 3.28774731e-05],
       [9.95145261e-01, 4.85474942e-03],
       [1.08196318e-01, 8.91803682e-01],
       [3.59025478e-01, 6.40974522e-01],
       [1.10499725e-01, 8.89500201e-01],
       [2.25538373e-01, 7.74461687e-01],
       [9.98749137e-01, 1.25089986e-03],
       [1.90371588e-01, 8.09628427e-01],
       [9.37200725e-01, 6.27992526e-02],
       [9.99367177e-01, 6.32850570e-04],
       [9.99963284e-01, 3.66732384e-05],
       [9.48564351e-01, 5.14356568e-02],
       [9.56201553e-01, 4.37984206e-02],
       [6.51602149e-02, 9.34839785e-01],
       [9.99999046e-01, 1.00051784e-06],
       [9.98620391e-01, 1.37962191e-03],
       [9.47624370e-02, 9.05237496e-01],
       [7.12222219e-01, 2.87777781e-01],
       [4.08830613e-01, 5.91169357e-01],
       [4.01257932e-01, 5.98742008e-01],
       [9.99981642e-01, 1.84092059e-05],
       [9.86927807e-01, 1.30722430e-02],
       [9.73069012e-01, 2.69310307e-02],
       [9.92525339e-01, 7.47464644e-03],
       [4.22180533e-01, 5.77819526e-01],
       [3.74090314e-01, 6.25909686e-01],
       [9.00554836e-01, 9.94452089e-02],
       [9.96229827e-01, 3.77019821e-03],
       [8.69540453e-01, 1.30459592e-01],
       [9.18236852e-01, 8.17631558e-02],
       [6.71503171e-02, 9.32849705e-01],
       [2.67355323e-01, 7.32644677e-01],
       [9.99998689e-01, 1.25849306e-06],
       [9.99991894e-01, 8.13550105e-06],
       [9.63819861e-01, 3.61801200e-02],
       [1.12400115e-01, 8.87599885e-01],
       [8.96893084e-01, 1.03106916e-01],
       [9.99994040e-01, 5.97917688e-06],
       [9.99433100e-01, 5.66851115e-04],
       [9.99959230e-01, 4.08172818e-05],
       [9.99471962e-01, 5.28042321e-04],
       [1.00907192e-01, 8.99092853e-01],
       [7.78602958e-01, 2.21397042e-01],
       [9.42606330e-01, 5.73936440e-02],
       [9.56334770e-02, 9.04366493e-01],
       [7.15143263e-01, 2.84856737e-01],
       [3.28363180e-01, 6.71636820e-01],
       [1.33568943e-01, 8.66431057e-01],
       [7.44434819e-02, 9.25556600e-01],
       [9.26322997e-01, 7.36770183e-02],
       [9.69936788e-01, 3.00631654e-02],
       [4.12148356e-01, 5.87851644e-01],
       [9.98996079e-01, 1.00392464e-03],
       [9.97383654e-01, 2.61637894e-03],
       [9.99999642e-01, 3.66045072e-07],
       [8.30568254e-01, 1.69431791e-01],
       [1.25899151e-01, 8.74100864e-01],
       [6.11705780e-02, 9.38829422e-01],
       [9.99969840e-01, 3.01911859e-05],
       [9.97462153e-01, 2.53788382e-03],
       [8.31490874e-01, 1.68509126e-01],
       [2.56192029e-01, 7.43807971e-01],
       [9.99974728e-01, 2.52405061e-05],
       [7.60781288e-01, 2.39218742e-01],
       [5.35193384e-01, 4.64806587e-01],
       [9.99709666e-01, 2.90352647e-04],
       [2.36094102e-01, 7.63905883e-01],
       [9.99740064e-01, 2.59921653e-04],
       [9.99999285e-01, 6.80445339e-07],
       [1.00000000e+00, 8.55684501e-09],
       [9.99842644e-01, 1.57280505e-04],
       [9.69608068e-01, 3.03919483e-02],
       [9.99734938e-01, 2.65098090e-04],
       [8.61145318e-01, 1.38854623e-01],
       [8.84838045e-01, 1.15161963e-01],
       [1.26957521e-01, 8.73042524e-01],
       [1.08776897e-01, 8.91223073e-01],
       [9.85367537e-01, 1.46324970e-02],
       [9.99420404e-01, 5.79623622e-04],
       [2.22981453e-01, 7.77018547e-01],
       [1.97749451e-01, 8.02250564e-01],
       [9.99945760e-01, 5.42549496e-05],
       [2.38410935e-01, 7.61589050e-01],
       [9.18569207e-01, 8.14308450e-02],
       [9.98411298e-01, 1.58868567e-03],
       [1.78226024e-01, 8.21774006e-01],
       [1.95776328e-01, 8.04223716e-01]], dtype=float32)
import numpy as np
y_pred = np.argmax(y_pred, axis=1)
y_pred
array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
       1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
       1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0,
       1, 1], dtype=int64)
 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model('model_vgg19.h5')
 
img=image.load_img('Dataset/Test/Uninfected/2.png',target_size=(224,224))
x=image.img_to_array(img)
x
array([[[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        ...,
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],

       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        ...,
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],

       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        ...,
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],

       ...,

       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        ...,
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],

       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        ...,
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],

       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        ...,
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]], dtype=float32)
x.shape
(224, 224, 3)
x=x/255
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape
(1, 224, 224, 3)
model.predict(img_data)
array([[0.01155142, 0.98844856]], dtype=float32)
a=np.argmax(model.predict(img_data), axis=1)
if(a==1):
    print("Uninfected")
else:
    print("Infected")
Uninfected


# In[ ]:




