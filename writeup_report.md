



```python



```

    Using TensorFlow backend.


## Model Architecture

Underlying model for this behavioural cloning project is a variant of LeNet architecture.

Using Keras to visualize model architecture:


```python

from keras.utils import plot_model
from keras import models
from keras.utils.vis_utils import model_to_dot
model= models.load_model('model.h5')
plot_model(model, to_file='images/model.png', show_shapes=True, show_layer_names=True)

# alternatively, use graphviz
# from IPython.display import SVG
# SVG(model_to_dot(model).create(prog='dot', format='svg'))

```

Model architecture

![model_architecture](images/model.png)


```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python

clip = (VideoFileClip("video.mp4")
        .resize(2))
clip.write_videofile("images/video.gif", fps=15, codec='gif')

```

    [MoviePy] >>>> Building video images/video.gif
    [MoviePy] Writing video images/video.gif


    100%|██████████| 1726/1726 [00:11<00:00, 153.08it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: images/video.gif 
    


    <img src="/images/model.gif" model>


```python

```
