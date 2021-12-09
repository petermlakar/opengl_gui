# opengl_gui
A simple graphical user interface built on Python and OpenGL

To install the library execute
```console
pip3 install PyOpenGL
pip3 install PyOpenGL_accelerate 
pip3 install glfw 
pip3 install numpy
pip3 install opencv-python
pip3 install freetype-py
pip3 install CairoSVG
pip3 install .
```
inside the base directory. The library can then be included in a project with
```
from opengl_gui import gui_components
from opengl_gui import gui_helper
```

# Example

## Initialization

To start using the library inside your project you must first initialize the OpenGL 
and GLFW instances. This is done by initializing the ``` Gui``` object from the 
components module

```python
from glfw import swap_buffers

from opengl_gui.gui_components import *
from opengl_gui.gui_helper     import *

gui = Gui(fullscreen = FULLSCREEN, width = WIDTH, height = HEIGHT)
```

The ``` FULLSCREEN, WIDTH, HEIGHT ``` variables denote a boolean flag and two integers
respectively. ``` FULLSCREEN``` is set to ``` true ``` if the application should be 
displayed in fullscreen mode. The other two are the window width and height in pixels.

## Simple scene creation

After the context has been initialized we can create a simple scene to be rendered. 
The simplest scene is just a basic container. Indeed it is recomended to start with
the ``` Container ``` component.

```python
from glfw import swap_buffers

from opengl_gui.gui_components import *
from opengl_gui.gui_helper     import *

gui = Gui(fullscreen = FULLSCREEN, width = WIDTH, height = HEIGHT)

simple_scene = Container(position = [0.0, 0.0], scale = [1.0, 1.0], colour = [1.0, 1.0, 1.0, 1.0], id = "simple_container")
simple_scene.update_geometry(parent = None)
```
We created the container component, a square with its <b>top left corner aligned to the
top left corner of the display</b>. It <b>covers the whole screen</b> since its scale is ```1.0```
in both axis. After the scene is created a call to  ``` update_geometry ``` on the root 
component is necessary. The root component in this case is the container, since it is the
only component not is depending on other components (hence its parent is None, it has
no parent component). This method computes the final position and scale of <b>this components 
 and all components that depend on it</b>.

## Dependence

Let us add a second component that depends on the ``` simple_scene ``` component.

```python
from glfw import swap_buffers

from opengl_gui.gui_components import *
from opengl_gui.gui_helper     import *

gui = Gui(fullscreen = FULLSCREEN, width = WIDTH, height = HEIGHT)

simple_scene = Container(position = [0.0, 0.0], scale = [1.0, 1.0], colour = [1.0, 1.0, 1.0, 1.0], id = "simple_container")
dependent_component = Container(position = [0.5, 0.5], scale = [0.5, 0.5], colour = [1.0, 0.0, 0.0, 1.0], id = "simple_container")

dependent_component.depends_on(element = simple_scene)

simple_scene.update_geometry(parent = None)
```

This will create a scene that will display a white background with a red square occupying
the bottom right quarter. The ``` depends_on ``` method creates a positional and scale 
dependence between ``` simple_scene ``` and ``` dependent_component ``` where the later
depends on the former. This entails that the position argument passed to its constructor
computes the <b>position in relative coordinates to the parent</b>. This means that a
position of ``` [0.0, 0.0] ``` in ``` dependent_component ``` corresponds to it being
positioned with its left top corrent aligned with ``` simple_scene ``` top left corner.
A position of ``` [1.0, 1.0] ``` corresponds to bottom right corner alignment. The scale
always remains <b>relative with respect to the main window size</b>.

## Rendering 

After a scene is constructed we can render it
```python
from glfw import swap_buffers

from opengl_gui.gui_components import *
from opengl_gui.gui_helper     import *

gui = Gui(fullscreen = FULLSCREEN, width = WIDTH, height = HEIGHT)

simple_scene = Container(position = [0.0, 0.0], scale = [1.0, 1.0], colour = [1.0, 1.0, 1.0, 1.0], id = "simple_container")
dependent_component = Container(position = [0.5, 0.5], scale = [0.5, 0.5], colour = [1.0, 0.0, 0.0, 1.0], id = "simple_container")

dependent_component.depends_on(element = simple_scene)

simple_scene.update_geometry(parent = None)

glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

simple_scene.execute(parent = None, gui = gui, custom_data = None)
        
swap_buffers(gui.window)
```

The OpenGL function ``` glClear ``` clears the depth and color buffers before rendering
is executed. If this command would be missing consequtive renders would overlay on top
of each other. The call to the method ``` execute ``` logically updates and renders the
the components in a hierarchical manner. Multiple dependent components added to the same
parent component are <b>renderer in the order they were added</b>.

## Main loop
