import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from math import sin, pi
import glfw
import numpy as np
import time
import cv2

from typing import Callable, Tuple
from opengl_gui.gui_shaders import *

class Gui():

    def __init__(self,
        fullscreen: bool = False,
        width:  int = 1920,
        height: int = 1080) -> None:

        if not glfw.init():
            print("Error initializing glfw...")
            exit()

        self.shader_pack = None
        self.geometry    = None

        self.active_shader = "none"
        self.active_shader_program = None

        self.x_pos = -1.0
        self.y_pos = -1.0

        self.x_pos_old = None
        self.y_pos_old = None

        self.dx = 0.0
        self.dy = 0.0

        self.mouse_in_window = 0
        self.mouse_press_event = (False, -1.0, -1.0)
        self.mouse_press_event_stack = []

        self.resize_event = None

        self.frames = 0

        self.time_fps = time.time()
        self.time_from_start = time.time()

        ####

        self.window_values = glfw.get_video_mode(glfw.get_primary_monitor())
        self.window_width  = self.window_values.size.width
        self.window_height = self.window_values.size.height

        self.fullscreen = fullscreen and width == self.window_width and height == self.window_height

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        self.width  = width  if width  <= 3840 else 3840
        self.height = height if height <= 2160 else 2160

        self.window = glfw.create_window(width, height, "VICOS Demo 1.5", glfw.get_primary_monitor() if self.fullscreen else None, None)
        self.window_aspect_ratio = self.width/self.height

        glfw.make_context_current(self.window)
        glfw.set_window_size_limits(self.window, 100, 100, 3850, 2160)
        glfw.set_cursor_pos_callback(self.window,       self.mouse_position_callback)
        glfw.set_cursor_enter_callback(self.window,     self.mouse_enter_callback)
        glfw.set_mouse_button_callback(self.window,     self.mouse_event_callback)

        glfw.set_framebuffer_size_callback(self.window, self.resize_event_callback)

        glfw.set_key_callback(self.window, self.key_press_event_callback)
     
        glfw.swap_interval(1)

        glEnable(GL_BLEND)
        #glEnable(GL_DEPTH_TEST)
        #glDepthFunc(GL_LESS)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.3, 0.3, 0.3, 1.0)

        texture_bgr_shader = TextureShaderBGR().compile()
        texture_r_shader   = TextureShaderR().compile()
        loading_shader     = LoadingShader().compile()
        default_shader = DefaultShader().compile()
        text_shader    = TextShader().compile()
        circle_shader  = CircleShader().compile()

        self.shader_pack = {
            "default_shader" : default_shader.generate_uniform_functions(), 
            "text_shader"    : text_shader.generate_uniform_functions(),
            "texture_shader" : texture_bgr_shader.generate_uniform_functions(),
            "texture_r_shader" : texture_r_shader.generate_uniform_functions(),
            "loading_shader"   : loading_shader.generate_uniform_functions(),
            "circle_shader" : circle_shader.generate_uniform_functions()}

        ####

        vertices = np.array(
            [1.0, -1.0, 1.0, 1.0,
             1.0,  1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 0.0,
            -1.0,  1.0, 0.0, 0.0,
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0], dtype=np.float32)

        self.vertex_buffer_array = glGenBuffers(1)
        self.number_of_vertices  = len(vertices)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer_array)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer_array)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, None)
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)

    def mouse_position_callback(self, window, x_pos, y_pos) -> None:

        self.x_pos = (x_pos/self.width)*2.0 - 1.0
        self.y_pos = -(y_pos/self.height)*2.0 + 1.0

        if self.x_pos_old is None:
            self.x_pos_old = self.x_pos
        if self.y_pos_old is None:
            self.y_pos_old = self.y_pos

        self.dx =  (self.x_pos - self.x_pos_old)/self.window_aspect_ratio
        self.dy = -(self.y_pos - self.y_pos_old)

        self.x_pos_old = self.x_pos
        self.y_pos_old = self.y_pos

    def mouse_enter_callback(self, window, entered) -> None:
        self.mouse_in_window = entered

    def mouse_event_callback(self, window, button, action, mods) -> None:

        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            self.mouse_press_event_stack.append([True, self.x_pos, self.y_pos])

        elif button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE: 
            self.mouse_press_event_stack.append([False, self.x_pos, self.y_pos])

    def resize_event_callback(self, window, width, height) -> None:

        self.resize_event = time.time()

        self.width  = width
        self.height = height 
        self.window_aspect_ratio = width/height

    def key_press_event_callback(self, window, key, scancode, action, mods) -> None:

        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
             glfw.set_window_should_close(window, glfw.TRUE)

    ####

    def draw(self):
        glDrawArrays(GL_TRIANGLES, 0, self.number_of_vertices)

    def switch_shader_program(self, shader: str):

        if shader != self.active_shader:
            self.active_shader_program = self.shader_pack[shader]
            glUseProgram(self.active_shader_program.shader_program)
            self.active_shader = shader

        return self.active_shader_program

    ####

    def poll_events(self):

        glfw.poll_events()

        self.mouse_press_event = self.mouse_press_event_stack.pop(0) if len(self.mouse_press_event_stack) > 0 else None

    def should_window_close(self):
        return glfw.window_should_close(self.window)

    def clear_screen(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def swap_buffers(self):

        glfw.swap_buffers(self.window)

        self.frames += 1
        self.dx = self.dy = 0.0

    def should_window_resize(self):

        if (self.resize_event is not None) and ((time.time() - self.resize_event) >= 0.5):

            glViewport(0, 0, self.width, self.height)
            self.resize_event = None

            return True

        return False

class Element():

    def __init__(self,
        position: list  = None,
        offset:   list  = None,
        scale:    list  = None,
        depth:    float = None,
        colour:   list = None,
        animations: dict = None,
        shader: str = None,
        id:     str = None):

        # Default initalizers for vairables

        self.position = np.array([0.0, 0.0], dtype = np.float32) if position is None else np.array(position, dtype = np.float32)
        self.offset   = np.array([0.0, 0.0], dtype = np.float32) if offset   is None else np.array(offset, dtype = np.float32)
        self.scale    = np.array([1.0, 1.0], dtype = np.float32) if scale    is None else np.array(scale, dtype = np.float32)

        self.properties = np.array([0.99 if depth is None else depth, 1.0], dtype = np.float32)
        self.colour     = np.array([1.0, 1.0, 1.0, 1.0], dtype = np.float32) if colour is None else np.array(colour, dtype = np.float32)
        self.animations = {} if animations is None else animations

        self.request_removal_array = []
        self.request_on_end        = []

        self.shader = "default_shader"  if shader is None else shader
        self.id     = "default_id"      if id     is None else id

        # Internal value initializations

        self.render_position = np.matrix([[1.0, 0.0, self.position[0] + self.scale[0]],
                                           [0.0, 1.0, self.position[1] - self.scale[1]],
                                           [0.0, 0.0, 1.0]])
        self.render_scale    = np.matrix([[self.scale[0], 0.0, 0.0], 
                                           [0.0, self.scale[1], 0.0], 
                                           [0.0, 0.0, 1.0]])
        self.transform       = np.matrix([[0.0, 0.0, 0.0], 
                                           [0.0, 0.0, 0.0], 
                                           [0.0, 0.0, 1.0]])

        self.top = None
        self.bot = None

        ####

        self.command_chain = []
        self.command_animations = []

        ####

        self.dependent_components = []

    #######################################################

    def execute(self, parent, gui, custom_data) -> None:
        
        for e in self.command_chain:
            e(parent = parent, gui = gui, custom_data = custom_data)

    def element_on_enter(self, parent, gui: Gui, custom_data) -> None:
        pass

    def element_update(self, parent, gui: Gui, custom_data) -> None:

        for anims in self.command_animations: # Only one animation at a time
            self.animations[anims].apply(element = self, custom_data = custom_data, request_on_end = self.request_on_end, request_removal_array = self.request_removal_array)
        
        if len(self.command_animations) > 0:
            self.update_geometry(parent = parent)

        for a in self.request_removal_array:
            self.command_animations.remove(a)
        for on_end in self.request_on_end:
            on_end(self, gui, custom_data)

        self.request_removal_array.clear()
        self.request_on_end.clear()

    def element_render(self, parent, gui: Gui, custom_data) -> None:
        pass

    def element_exit(self, parent, gui: Gui, custom_data) -> None:
        
        for c in self.dependent_components:
            c.execute(parent = self, gui = gui, custom_data = custom_data)

    #######################################################

    ### Geometry update functionality ###

    def update_geometry(self, parent) -> None:

        topLeft = parent.top if parent is not None else [-1.0, 1.0]

        self.render_position[0,2] = topLeft[0] + self.scale[0] + ((2*parent.scale[0]*self.position[0] + self.offset[0]) if parent is not None else 2*self.position[0])
        self.render_position[1,2] = topLeft[1] - self.scale[1] - ((2*parent.scale[1]*self.position[1] + self.offset[1]) if parent is not None else 2*self.position[1])
    
        self.render_scale[0,0] = self.scale[0]
        self.render_scale[1,1] = self.scale[1]

        self.transform = self.render_position*self.render_scale

        self.top = (self.transform[0,1] + self.transform[0,2] - self.transform[0,0],
                    self.transform[1,1] + self.transform[1,2] - self.transform[1,0])
        self.bot = (-self.transform[0,1] + self.transform[0,2] + self.transform[0,0],
                    -self.transform[1,1] + self.transform[1,2] + self.transform[1,0])

        for c in self.dependent_components:
            c.update_geometry(parent = self)

    ### Visual properties manipulation ###

    def is_inside(self, x: float, y: float) -> bool:
        return (x >= self.top[0]) and (y <= self.top[1]) and (x <= self.bot[0]) and (y >= self.bot[1])
    
    def get_aspect_ratio(self) -> float:
        return self.scale[0]/self.scale[1]

    def center_x(self):

        self.position[0] = 0.5
        self.offset[0] = -self.scale[0]

    def center_y(self):

        self.position[1] = 0.5
        self.offset[1] = -self.scale[1]

    def set_highlight(self, highlight: float) -> None:
        self.properties[1] = highlight

    def set_depth(self, depth: float) -> None:
        self.properties[0] = depth

    ### Animation manipulation ###

    def animation_play(self, animation_to_play: str):

        if (animation_to_play not in self.animations) or\
           (animation_to_play in self.command_animations):
           return

        self.animations[animation_to_play].play(element = self)
        self.command_animations.append(animation_to_play)

    def animation_stop(self, animation_to_stop: str):

        self.command_animations = [x for x in self.command_animations if x != animation_to_stop]

    ### Dependent component manipulation ###

    def depends_on(self, element) -> None:
        element.dependent_components.append(self)
        

class Animation():

    def __init__(self,
        transform: Tuple[str, list],
        id: str,
        on_end:    Callable = None,
        duration:  float    = 1.0):

        self.transform = transform
        self.duration  = duration
        self.on_end    = on_end
        self.id = id

        self.start = None
        self.start_initial_values = None

    def apply(self, element: Element, custom_data, request_on_end: list, request_removal_array: list) -> None:
        pass

    def apply_end(self, element: Element, custom_data) -> None:
        pass

    def play(self, element: Element) -> None:

        self.start = time.time()
        self.start_initial_values = getattr(element, self.transform[0]).copy()

    def stop(self, element: Element) -> None:
        self.start = None

    def interpolate_sin(self, start: list, goal: list, time: float):

        t0 = sin(time*pi*0.5)
        out = [0]*len(start)

        for i in range(0, len(start)):
            out[i] = start[i]*(1.0 - t0) + goal[i]*t0

        return out

class AnimationScalar(Animation):

    def __init__(self, transform: Tuple[str, list], id: str, on_end: Callable = None, duration: float = 1):
        super().__init__(transform = transform, on_end = on_end, duration = duration, id = id)

    def apply(self, element: Element, custom_data, request_on_end: list, request_removal_array: list) -> None:
        
        elapsed_time = time.time() - self.start

        if elapsed_time < self.duration:

            t0 = sin(0.5*pi*elapsed_time/self.duration)

            setattr(element, self.transform[0], self.start_initial_values*(1.0 - t0) + self.transform[1]*t0)
        else:

            setattr(element, self.transform[0], self.transform[1])

            self.start = None
            if self.on_end is not None:
                request_on_end.append(self.on_end)

            request_removal_array.append(self.id)
    
    def apply_end(self, element: Element, custom_data) -> None:
        
        setattr(element, self.transform[0], self.transform[1])
        self.start = None

class AnimationList(Animation):

    def __init__(self, 
        transform: Tuple[str, list],
        id: str,
        on_end:    Callable = None, 
        duration:  float = 1):
        super().__init__(transform = transform, on_end = on_end, duration = duration, id = id)

    def play(self, element: Element) -> None:

        self.start = time.time()
        self.start_initial_values = getattr(element, self.transform[0]).copy()

        for i in range(0, len(self.transform[1])):
            self.transform[1][i] = self.start_initial_values[i] if self.transform[1][i] is None else self.transform[1][i]

    def apply(self, element: Element, custom_data, request_on_end: list, request_removal_array: list) -> None:
        
        elapsed_time = time.time() - self.start

        if elapsed_time < self.duration:

            t0 = sin(0.5*pi*(elapsed_time/self.duration))

            i0 = self.start_initial_values
            i1 = self.transform[1]

            output = getattr(element, self.transform[0])

            for i in range(0, len(i0)):
                output[i] = i0[i]*(1.0 - t0) + i1[i]*t0
        else:

            output = getattr(element, self.transform[0])
            i1 = self.transform[1]

            for i in range(0, len(self.start_initial_values)):
                output[i] = i1[i]

            self.start = None
            if self.on_end is not None:
                request_on_end.append(self.on_end)

            request_removal_array.append(self.id)

    def apply_end(self, element: Element, custom_data) -> None:
        
        output = getattr(element, self.transform[0])
        i1 = self.transform[1]

        for i in range(0, len(self.start_initial_values)):
            output[i] = i1[i]

        self.start = None
        
class AnimationListOne(Animation):

    def __init__(self, transform: Tuple[str, list], id: str, index: int, on_end: Callable = None, duration: float = 1):
        super().__init__(transform = transform, on_end = on_end, duration = duration, id = id)

        self.index = index

    def apply(self, element: Element, custom_data, request_on_end: list, request_removal_array: list) -> None:
        
        elapsed_time = time.time() - self.start

        if elapsed_time < self.duration:

            t0 = sin(0.5*pi*elapsed_time/self.duration)

            i0 = self.start_initial_values
            i1 = self.transform[1]

            output = getattr(element, self.transform[0])
            output[self.index] = i0[self.index]*(1.0 - t0) + i1*t0
        else:

            output = getattr(element, self.transform[0])
            output[self.index] = self.transform[1]

            self.start = None
            if self.on_end is not None:
                request_on_end.append(self.on_end)

            request_removal_array.append(self.id)

    def apply_end(self, element: Element, custom_data) -> None:
        
        output = getattr(element, self.transform[0])
        output[self.index] = self.transform[1]

        self.start = None

class Video():

    def __init__(
        self, 
        path: str, 
        loop: bool = False):

        self.path = path

        self.video_path = path 
        self.video = cv2.VideoCapture(path)

        self.loop = loop

        self.last_frame_time = None

        self.frame_index = 0
        self.frame = cv2.cvtColor(self.video.read()[1], cv2.COLOR_BGR2RGB)
        self.f = 0

        self.play_toggle = False

        self.video_fps = self.video.get(cv2.CAP_PROP_FPS)
        self.n_frames  = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration  = self.n_frames/self.video_fps

    def get_frame(self):

        if self.video is None or \
           not self.play_toggle or \
           (not self.loop and self.frame_index == self.n_frames):
            return None

        d = (time.time() - self.last_frame_time)*self.video_fps
        if d >= 1.0:
            self.last_frame_time = time.time()
            self.f += d

        if self.frame_index <= self.n_frames:

            while 1.0 <= self.f:

                flag, self.frame = self.video.read()
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB) if flag else None
                self.frame_index += 1
                self.f = self.f - 1.0

            return self.frame
        else:
            if self.loop:

                self.last_frame_time = time.time()
                self.frame_index = 0
                self.f = 0

                self.video.release()
                self.video = cv2.VideoCapture(self.video_path)

                return None
            else:
                return None

    def reset_and_play(self):

        self.reset()
        self.play()

    def pause(self):

        self.play_toggle = False

    def resume(self):

        self.play_toggle = True
        self.last_frame_time = time.time()
        self.f          = 0

    def play(self):

        self.play_toggle = True
        self.last_frame_time = time.time()
        self.frame_index = 0
        self.f = 0

    def reset(self):

        self.video.release()
        self.video = cv2.VideoCapture(self.video_path)

    def __del__(self):
        self.video.release()

###########################################################

class Container(Element):

    def __init__(self,  
        position: list, 
        scale:  list,
        id:     str,
        offset: list = None, 
        depth:  float = None, 
        colour: list  = None, 
        animations: dict = None, 
        shader: str = "default_shader"):

        super(Container, self).__init__(
            position = position,
            offset   = offset,
            scale    = scale,
            depth    = depth,
            colour   = colour,
            animations = animations,
            shader = shader,
            id     = id)

        self.command_chain.append(self.element_update)
        self.command_chain.append(self.element_render)
        self.command_chain.append(self.element_exit)

    def element_render(self, parent, gui: Gui, custom_data) -> None:

        shader_program = gui.switch_shader_program(shader = self.shader)

        shader_program.uniform_functions["transform"](self.transform)
        shader_program.uniform_functions["colour"](self.colour)
        shader_program.uniform_functions["properties"](self.properties)
        #shader_program.uniform_functions["highlight"](self.highlight)
        #shader_program.uniform_functions["depth"](self.depth)

        gui.draw()

class Char(Element):

    def __init__(self,  
        position: list,
        scale:    list,
        id: str,
        texture: int,
        offset: list = None,
        depth:  float = None, 
        colour: list  = None,
        shader: str = "text_shader"):

        super(Char, self).__init__(
            position = position,
            offset   = offset,
            scale    = scale,
            depth    = depth,
            colour   = colour,
            shader = shader,
            id     = id)

        self.texture = texture

        self.command_chain.append(self.element_render)

    #######################################################

    def element_render(self, parent, gui: Gui, custom_data) -> None:
        
        shader_program = gui.switch_shader_program(shader = self.shader)
        shader_program.uniform_functions["transform"](self.transform)
        shader_program.uniform_functions["colour"](self.colour)
        shader_program.uniform_functions["depth"](self.properties[0])

        glBindTexture(GL_TEXTURE_2D, self.texture)

        gui.draw()

    def is_inside(self, x: float, y: float) -> bool:
        return False

class TextField(Element):

    def __init__(
        self, 
        position:   list,
        text_scale: int,
        id: str, 
        aspect_ratio: float,
        offset: list = None, 
        animations: dict = None,
        depth:  float = None,
        colour: list  = None, 
        shader: str = "text_shader",
        update: Callable = None, 
        width:  int      = 1920):

        super(TextField, self).__init__(
            position = position,
            offset   = offset,
            scale    = None,
            depth    = depth,
            colour   = colour,
            shader = shader,
            animations = animations,
            id     = id)

        self.update = update
        self.width  = width
        self.aspect_ratio = aspect_ratio
        self.text_scale = text_scale 

        self.command_chain.append(self.element_exit)

    def set_text(self, font: dict, text: str):
        
        self.scale_x = 0.0
        self.scale_y = 0.0
        space = np.int(np.ceil((font["size"] >> 6)*0.20))

        init_offset = font[list(text)[0]]["bearing"][0]
        x = -init_offset if init_offset < 0 else 0

        chars = []

        for c in list(text):
            
            if c == ' ':
                x += space
                self.scale_x += space*self.text_scale/self.width
                continue

            sx = font[c]["size"][0]
            sy = font[c]["size"][1]*self.aspect_ratio

            chars.append((x + font[c]["bearing"][0], 
                - font[c]["bearing"][1]*self.aspect_ratio,
                sx,
                sy,
                font[c]["texture"]))

            adv = (font[c]["advance"][0] >> 6)
            x  += adv

            self.scale_x += adv*self.text_scale/self.width

        chars = list(map(lambda i: (self.text_scale*i[0]/self.width, 
                                    self.text_scale*i[1]/self.width, 
                                    self.text_scale*i[2]/self.width, 
                                    self.text_scale*i[3]/self.width, i[4]), chars))

        self.scale_y = chars[0][3]

        self.dependent_components.clear()
        for c in chars:

            t = Char(
                colour   = self.colour,
                position = [c[0], c[1]],
                scale    = [c[2], c[3]],
                texture  = c[4],
                depth = self.properties[0],
                id = "Char")
            
            t.depends_on(element = self)

        self.update_geometry(parent = None)

    #######################################################

    def is_inside(self, x: float, y: float) -> bool:
        return False

    def center_x(self):
        
        self.position[0] = 0.5
        self.offset[0]   = -self.scale_x

    def center_y(self):
        
        self.position[1] = 0.5
        self.offset[1]   = self.scale_y

class TextureRGB(Element):

    def __init__(self, 
        position: list,
        scale: list,
        alpha: float,
        id:     str,
        get_texture: Callable,
        offset: list  = None, 
        depth:  float = None, 
        animations: dict  = None, 
        shader:   str = "texture_shader", 
        set_once: bool = False):

        super().__init__(
            position = position, 
            offset   = offset, 
            scale    = scale, 
            depth    = depth,
            animations = animations, 
            shader = shader, 
            id = id)

        self.get_texture = get_texture
        self.set_once    = set_once

        self.texture = None

        assert(self.get_texture is not None)

        self.command_chain.append(self.texture_initialization)

        self.properties[1] = alpha

    ###########################################################

    def element_render(self, parent, gui: Gui, custom_data) -> None:
        
        shader_program = gui.switch_shader_program(shader = self.shader)
        shader_program.uniform_functions["transform"](self.transform)
        shader_program.uniform_functions["properties"](self.properties)

        glBindTexture(GL_TEXTURE_2D, self.texture)
        gui.draw()

    def set_alpha(self, alpha: float) -> None:
        self.properties[1] = alpha

    ###########################################################

    def texture_initialization(self, parent, gui: Gui, custom_data) -> None:

        texture_buffer = self.get_texture(gui, custom_data) # TODO Make sure this is standard practice

        if texture_buffer is None:
            return

        self.texture = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_buffer.shape[1], texture_buffer.shape[0],
                0, GL_RGB, GL_UNSIGNED_BYTE, texture_buffer)
        glGenerateTextureMipmap(self.texture)

        if self.set_once:
            self.command_chain = [self.element_update, self.element_render]
        else:
            self.command_chain = [self.texture_update, self.element_update, self.element_render]

    def texture_update(self, parent, gui: Gui, custom_data) -> None:

        texture_buffer = self.get_texture(gui, custom_data)

        if texture_buffer is None:
            return

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_buffer.shape[1], texture_buffer.shape[0],
                0, GL_RGB, GL_UNSIGNED_BYTE, texture_buffer)
        glGenerateTextureMipmap(self.texture)

    def __del__(self):
        if self.texture is not None:

            glDeleteTextures(1, self.texture)

            self.texture    = None
            self.get_texture = None

class TextureR(Element):

    def __init__(self, 
        position: list, 
        scale:  list, 
        colour: list,
        id:     str,
        get_texture: Callable,
        offset: list  = None, 
        depth:  float = None,
        animations: dict  = None, 
        shader: str = "texture_r_shader", 
        set_once: bool = False):

        super().__init__(
            position = position, 
            offset   = offset, 
            scale    = scale, 
            depth    = depth,
            colour   = colour,
            animations = animations, 
            shader = shader, 
            id = id)

        self.get_texture = get_texture
        self.set_once    = set_once

        self.texture = None

        assert(self.get_texture is not None)

        self.command_chain.append(self.texture_initialization)
    
    def element_render(self, parent, gui: Gui, custom_data) -> None:

        shader_program = gui.switch_shader_program(shader = self.shader)
        shader_program.uniform_functions["transform"](self.transform)
        shader_program.uniform_functions["colour"](self.colour)

        glBindTexture(GL_TEXTURE_2D, self.texture)
        gui.draw()

    ###########################################################

    def texture_initialization(self, parent, gui: Gui, custom_data) -> None:

        texture_buffer = self.get_texture(gui, custom_data) # TODO Make sure this is standard practice

        if texture_buffer is None:
            return

        self.texture = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, texture_buffer.shape[1], texture_buffer.shape[0],
                0, GL_RED, GL_UNSIGNED_BYTE, texture_buffer)
        glGenerateTextureMipmap(self.texture)

        if self.set_once:
            self.command_chain = [self.element_update, self.element_render]
        else:
            self.command_chain = [self.texture_update, self.element_update, self.element_render]

    def texture_update(self, parent, gui: Gui, custom_data) -> None:

        texture_buffer = self.get_texture(gui, custom_data)

        if texture_buffer is None:
            return

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, texture_buffer.shape[1], texture_buffer.shape[0],
                0, GL_RED, GL_UNSIGNED_BYTE, texture_buffer)
        glGenerateTextureMipmap(self.texture)

    def __del__(self):
        if self.texture is not None:

            glDeleteTextures(1, self.texture)

            self.texture    = None
            self.get_texture = None

class DisplayTexture(Element):

    def __init__(self,  
        position: list, 
        scale:    list,
        id:     str,
        get_texture: Callable,
        offset:   list = None,
        depth:    float = None,
        aspect:   float = 1920.0/1080.0,
        colour:   list  = None, 
        animations: dict = None, 
        shader: str = "loading_shader"):

        super(DisplayTexture, self).__init__(
            position = position,
            offset   = offset,
            scale    = scale,
            depth    = depth,
            colour   = colour,
            animations = animations,
            shader = shader,
            id     = id)

        self.texture = None

        self.aspect = aspect
        self.stage_time = 0
        self.get_texture = get_texture
        self.aspect = aspect

        assert(self.get_texture is not None)

        self.stage_transition_duration = 1.0

        self.command_chain.append(self.element_update_stage_0)
        self.command_chain.append(self.element_render_stage_0)

    def element_render_stage_0(self, parent, gui: Gui, custom_data) -> None:

        shader_program = gui.switch_shader_program(shader = self.shader)

        now = time.time()

        time_start = now - gui.time_from_start

        shader_program.uniform_functions["stage"](0)
        shader_program.uniform_functions["depth"](self.properties[0])
        shader_program.uniform_functions["transform"](self.transform)
        shader_program.uniform_functions["animation_details"]([time_start, self.aspect, 0.0, self.stage_transition_duration])

        gui.draw()

    def element_render_stage_1(self, parent, gui: Gui, custom_data) -> None:


        shader_program = gui.switch_shader_program(shader = self.shader)

        now = time.time()

        time_start   = now - gui.time_from_start
        time_stage_0 = now - self.stage_time

        shader_program.uniform_functions["stage"](1)
        shader_program.uniform_functions["depth"](self.properties[0])
        shader_program.uniform_functions["transform"](self.transform)
        shader_program.uniform_functions["animation_details"]([time_start, self.aspect, time_stage_0, self.stage_transition_duration])

        glBindTexture(GL_TEXTURE_2D, self.texture)

        gui.draw()

    def element_update_stage_0(self, parent, gui: Gui, custom_data) -> None:
        
        super().element_update(parent = parent, gui = gui, custom_data = custom_data)

        texture_buffer = self.get_texture(gui, custom_data)

        if texture_buffer is None:
            return

        self.texture = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_buffer.shape[1], texture_buffer.shape[0],
                0, GL_RGB, GL_UNSIGNED_BYTE, texture_buffer)

        self.stage_time = time.time()

        self.command_chain = [self.element_update_stage_1, self.element_render_stage_1, self.element_exit]

    def element_update_stage_1(self, parent, gui: Gui, custom_data) -> None:

        super().element_update(parent = parent, gui = gui, custom_data = custom_data)

        texture_buffer = self.get_texture(gui, custom_data)

        if texture_buffer is None:
            return

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_buffer.shape[1], texture_buffer.shape[0],
                0, GL_RGB, GL_UNSIGNED_BYTE, texture_buffer)

    def __del__(self):

        if self.texture is not None:

            glDeleteTextures(1, self.texture)

            self.texture    = None
            self.getTexture = None

class Button(Element):

    def __init__(self,  
        position: list, 
        scale:    list,
        id:     str,
        offset:   list = None,
        depth:    float = None, 
        colour:   list  = None, 
        animations: dict = None, 
        shader: str = "default_shader",
        on_click: Callable = None):

        super(Button, self).__init__(
            position = position,
            offset   = offset,
            scale    = scale,
            depth    = depth,
            colour   = colour,
            animations = animations,
            shader = shader,
            id     = id)

        self.on_click = on_click

        self.mouse_click_count = 0
        self.mouse_press = False

        self.command_chain.append(self.element_update)
        self.command_chain.append(self.element_render)
        self.command_chain.append(self.element_exit)

    def element_update(self, parent, gui: Gui, custom_data) -> None:

        super().element_update(parent = parent, gui = gui, custom_data = custom_data)

        self.mouse_hover = gui.mouse_in_window and self.is_inside(gui.x_pos, gui.y_pos)
        self.properties[1] = 1.5 if self.mouse_hover else 1.0

        self.check_for_click(gui = gui, custom_data = custom_data)

    def element_render(self, parent, gui: Gui, custom_data) -> None:
        
        shader_program = gui.switch_shader_program(shader = self.shader)

        shader_program.uniform_functions["transform"](self.transform)
        shader_program.uniform_functions["properties"](self.properties)
        shader_program.uniform_functions["colour"](self.colour)

        gui.draw()

    def check_for_click(self, gui: Gui, custom_data) -> None:

        if gui.mouse_press_event is not None:

            mouse_press_event = gui.mouse_press_event

            inside = self.is_inside(mouse_press_event[1], mouse_press_event[2])

            if inside:

                if mouse_press_event[0]:
                    self.mouse_press = True
                elif not mouse_press_event[0] and self.mouse_press:

                    self.mouse_click_count += 1
                    self.mouse_press = False

                    if self.on_click is not None:
                        self.on_click(self, gui, custom_data)

                gui.mouse_press_event = None
            else:

                if self.mouse_press and mouse_press_event[0]:
                    self.mouse_press = False

                    gui.mouse_press_event = None

    def set_colour(self, colour: list) -> None:
        self.colour = np.array(colour, dtype = np.float32)

class DrawerMenu(Element):

    def __init__(self,  
        position: list,
        scale:    list,
        id:       str,
        position_closed: list,
        position_opened: list,
        offset:   list = None,  
        depth:    float = None, 
        colour:   list  = None, 
        animations: dict = None, 
        shader: str = "default_shader",
        on_open:  Callable = None,
        on_close: Callable = None,
        on_grab:  Callable = None):

        super(DrawerMenu, self).__init__(
            position = position,
            offset   = offset,
            scale    = scale,
            depth    = depth,
            colour   = colour,
            animations = animations,
            shader = shader,
            id     = id)

        self.position_opened = np.array(position_opened, dtype = np.float32)
        self.position_closed = np.array(position_closed, dtype = np.float32)
    
        self.v = self.position_opened - self.position_closed
        self.v_norm = self.v[0]**2 + self.v[1]**2

        self.opening = False

        self.open = self.grab = False
        self.open_event_flag  = False
        self.grab_event_flag  = False

        self.transition_duration = 0.25
        self.transition_position = None
        self.transition_release_time = None

        self.on_grab  = on_grab
        self.on_open  = on_open
        self.on_close = on_close

        self.command_chain.append(self.element_update)
        self.command_chain.append(self.element_exit)

    def element_update(self, parent, gui: Gui, custom_data) -> None:
        
        self.check_for_grab(gui = gui)

        if self.grab and (gui.dx**2 + gui.dy**2) != 0:

            self.old_position = self.position.copy()

            self.position += (self.v[0]*gui.dx + self.v[1]*gui.dy)/(self.v_norm)*self.v

            v_new = (self.position_opened - self.position)/self.v_norm
            v_new = self.v[0]*v_new[0] + self.v[1]*v_new[1]

            self.position = self.position_opened.copy() if v_new < 0.0 else (self.position_closed.copy() if np.abs(v_new) > 1.0 else self.position)

            self.update_geometry(parent = parent)
        elif not self.grab:
            self.apply_motion(parent = parent)

        if self.open_event_flag:

            if self.on_open is not None and self.open:
                self.on_open(self, gui)
            elif self.on_close is not None and not self.open:
                self.on_close(self, gui)

            self.open_event_flag = False

        elif self.grab_event_flag:

            if self.on_grab is not None:
                self.on_grab(self, gui)

            self.grab_event_flag = False

    def check_for_grab(self, gui: Gui) -> None:

        if gui.mouse_press_event is not None:

            mouse_press_event = gui.mouse_press_event

            inside = self.is_inside(mouse_press_event[1], mouse_press_event[2])

            if inside and mouse_press_event[0]: # Grabbed
                
                self.grab = True

                self.opening = False
                self.transition_release_time = self.transition_position = None

                self.grab_event_flag = True
            elif not mouse_press_event[0] and self.grab: # Release grab

                self.grab = False
                self.transition_release_time = time.time()
                self.transition_position = self.position.copy()

                # Determine if drawer should be closing or opening
                # after release.

                d = 0.1**2

                dist_to_open = self.position_opened - self.position
                dist_to_open = (dist_to_open[0]**2 + dist_to_open[1]**2)/self.v_norm

                dist_to_closed = self.position_closed - self.position
                dist_to_closed = (dist_to_closed[0]**2 + dist_to_closed[1]**2)/self.v_norm

                self.opening = dist_to_open < d if self.open else dist_to_closed > d

    def apply_motion(self, parent) -> None:
        
        if self.transition_release_time is not None:

            t = (time.time() - self.transition_release_time)/self.transition_duration
            i = sin(t*pi*0.5)

            p = self.position_opened if self.opening else self.position_closed

            self.position = (self.transition_position*(1.0 - i) + p*i).copy()

            if t >= 1.0:

                self.position = self.position_opened.copy() if self.opening else self.position_closed.copy()

                self.open     = self.opening
                self.open_event_flag = True

                self.opening = False
                self.transition_position = self.transition_release_time = None

            self.update_geometry(parent = parent)

class DemoDisplay(Element):

    def __init__(self,  
        position: list,
        scale:    list,
        id:       str,
        offset:   list = None,  
        depth:    float = None, 
        colour:   list  = None, 
        shader: str = "default_shader"):

        super(DemoDisplay, self).__init__(
            position = position,
            offset   = offset,
            scale    = scale,
            depth    = depth,
            colour   = colour,
            shader = shader,
            id     = id)

        self.default = None

        self.active_demo = None
        self.active_demo_button = None

        self.active_video = None
        self.active_video_button = None

        self.in_transition_demo  = []
        self.in_transition_video = []

    def insert_default(self, element: Element):
        self.default = element

    def insert_active_demo(self, active_demo: Element, active_demo_button: Button):

        self.remove_active_demo()

        self.active_demo        = active_demo
        self.active_demo_button = active_demo_button

        self.active_demo.animation_play(animation_to_play = "in")

    def insert_active_video(self, active_video: Element, active_video_button: Button):

        self.remove_active_video()

        self.active_video        = active_video
        self.active_video_button = active_video_button

        self.active_video.animation_play(animation_to_play = "in")

    def remove_active_demo(self):

        if self.active_demo is None:
            return

        def on_end(element, gui, custom_data):
            self.in_transition_demo.pop(0)

        self.active_demo.animations["out"].on_end = on_end
        self.active_demo.animation_play(animation_to_play = "out")

        self.in_transition_demo.append(self.active_demo)

        self.active_demo = None
        self.active_demo_button = None

    def remove_active_video(self):

        if self.active_video is None:
            return
        
        def on_end(element, gui, custom_data):
            self.in_transition_video.pop(0)

        self.active_video.animations["out"].on_end = on_end
        self.active_video.animation_play(animation_to_play = "out")

        self.in_transition_video.append(self.active_video)

        self.active_video = None
        self.active_video_button = None

    def execute(self, parent, gui: Gui, custom_data) -> None:
        
        shader_program = gui.switch_shader_program(shader = self.shader)

        shader_program.uniform_functions["transform"](self.transform)
        shader_program.uniform_functions["properties"](self.properties)
        shader_program.uniform_functions["colour"](self.colour)

        gui.draw()


        self.default.execute(parent = self, gui = gui, custom_data = custom_data)

        if self.active_demo is not None:
            self.active_demo.execute(parent = self, gui = gui, custom_data = custom_data)
        if self.active_video is not None:
            self.active_video.execute(parent = self, gui = gui, custom_data = custom_data)

        for t in self.in_transition_demo:
            t.execute(parent = self, gui = gui, custom_data = custom_data)
        for t in self.in_transition_video:
            t.execute(parent = self, gui = gui, custom_data = custom_data)

        for t in self.dependent_components:
            t.execute(parent = self, gui = gui, custom_data = custom_data)

    def update_geometry(self, parent) -> None:

        super().update_geometry(parent = parent)

        if self.default is not None:
            self.default.update_geometry(parent = self)
        if self.active_demo is not None:
            self.active_demo.update_geometry(parent = self)
        if self.active_video is not None:
            self.active_video.update_geometry(parent = self)

        for t in self.in_transition_demo:
            t.update_geometry(parent = self)
        for t in self.in_transition_video:
            t.update_geometry(parent = self)

class Circle(Element):

    def __init__(self,  
        position: list, 
        scale:    list,
        id: str,
        offset:   list = None,
        depth:    float = None, 
        colour:   list  = None, 
        animations: dict = None, 
        shader: str = "circle_shader"):

        super(Circle, self).__init__(
            position = position,
            offset   = offset,
            scale    = scale,
            depth    = depth,
            colour   = colour,
            animations = animations,
            shader = shader,
            id     = id)

    def execute(self, parent, gui: Gui, custom_data):

        shader_program = gui.switch_shader_program(shader = self.shader)

        shader_program.uniform_functions["transform"](self.transform)
        shader_program.uniform_functions["properties"](self.properties)
        shader_program.uniform_functions["colour"](self.colour)

        gui.draw()

class RangeSlider(Element):

    def __init__(self,  
        position: list, 
        scale:    list,
        range_bottom: int,
        range_top:    int,
        aspect_ratio: float,
        id: str,
        offset:   list = None,
        depth:    float = None, 
        colour:   list  = None, 
        animations: dict = None, 
        shader: str = "default_shader",
        on_select: Callable = None):

        super(RangeSlider, self).__init__(
            position = position,
            offset   = offset,
            scale    = scale,
            depth    = depth,
            colour   = colour,
            animations = animations,
            shader = shader,
            id     = id)

        self.range_top = range_top
        self.range_bottom = range_bottom
        self.selected_value = (range_top + range_bottom)*0.5

        self.on_select = on_select

        self.circle = Circle(
            position = [0.0, 0.0],
            scale    = [scale[0]*0.3/aspect_ratio, scale[0]*0.3],
            colour   = [1.0, 1.0, 1.0, 1.0],
            id = "slider_circle")

        self.circle.center_x()
        self.circle.center_y()

        self.circle.depends_on(element = self)

    def execute(self, parent, gui: Gui, custom_data):
        
        shader_program = gui.switch_shader_program(shader = self.shader)

        shader_program.uniform_functions["transform"](self.transform)
        shader_program.uniform_functions["properties"](self.properties)
        shader_program.uniform_functions["colour"](self.colour)

        gui.draw()

        for t in self.dependent_components:
            t.execute(parent = self, gui = gui, custom_data = custom_data)

