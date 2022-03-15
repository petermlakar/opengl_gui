from OpenGL.GL import *
from OpenGL.GL import shaders

class ShaderProgram():

    def __init__(self):
        self.vs = None
        self.fs = None

        self.uniform_functions = {}

    def compile(self):
        self.shader_program = shaders.compileProgram(shaders.compileShader(self.vs, GL_VERTEX_SHADER), shaders.compileShader(self.fs, GL_FRAGMENT_SHADER))

        return self

class DefaultShader(ShaderProgram):

    def __init__(self):
        super(DefaultShader, self).__init__()

        self.vs = \
        """
            #version 450 
            uniform mat3 transform;
            uniform vec2 properties;

            layout (location = 0) in vec2 position;
            out float highlight;
            void main()
            {
                highlight = properties[1];
                vec3 tmp = transform*vec3(position,1.0);
                gl_Position = vec4(tmp.x, tmp.y, 0.0, tmp.z);
            }
        """

        self.fs = \
        """
            #version 450 
            uniform vec4 colour;
            in float highlight;

            out vec4 fragColour;

            void main()
            {
                fragColour = colour;
                fragColour[0] = clamp(fragColour[0]*highlight, 0.0, 1.0);
                fragColour[1] = clamp(fragColour[1]*highlight, 0.0, 1.0);
                fragColour[2] = clamp(fragColour[2]*highlight, 0.0, 1.0);
            }
        """

    def generate_uniform_functions(self):

        self.transform_uniform_location = glGetUniformLocation(self.shader_program, "transform")
        self.colour_uniform_location = glGetUniformLocation(self.shader_program, "colour")
        self.properties_uniform_location = glGetUniformLocation(self.shader_program, "properties")
        

        self.uniform_functions["transform"] = lambda M: glUniformMatrix3fv(self.transform_uniform_location, 1, GL_TRUE, M)
        self.uniform_functions["colour"]    = lambda C: glUniform4fv(self.colour_uniform_location, 1, C)
        self.uniform_functions["properties"] = lambda P: glUniform2fv(self.properties_uniform_location, 1, P)


        return self

class TextShader(ShaderProgram):

    def __init__(self):
        super(TextShader, self).__init__()

        self.vs = \
        """
            #version 450 
            uniform mat3 transform;
            uniform float depth;

            layout (location = 0) in vec2 position;
            layout (location = 1) in vec2 texCoordIn;
            out vec2 texCoordOut;
            void main()
            {
                vec3 tmp = transform*vec3(position,1.0);
                gl_Position = vec4(tmp.x,tmp.y,0.0,tmp.z);
                texCoordOut = texCoordIn;
            }
        """

        self.fs = \
        """
            #version 450 
            uniform vec4 colour;
            uniform sampler2D text;

            in vec2 texCoordOut;
            out vec4 fragColour;

            void main()
            {
                fragColour = vec4(vec3(colour), texture(text, texCoordOut).r);
            }
        """

    def generate_uniform_functions(self):

        self.transform_uniform_location = glGetUniformLocation(self.shader_program, "transform")
        self.colour_uniform_location = glGetUniformLocation(self.shader_program, "colour")
        self.depth_uniform_location  = glGetUniformLocation(self.shader_program, "depth")

        self.uniform_functions["transform"] = lambda M: glUniformMatrix3fv(self.transform_uniform_location, 1, GL_TRUE, M)
        self.uniform_functions["colour"]    = lambda C: glUniform4fv(self.colour_uniform_location, 1, C)
        self.uniform_functions["depth"]    = lambda D: glUniform1f(self.depth_uniform_location, D)

        return self

class TextureShaderBGR(ShaderProgram):

    def __init__(self):
        super(TextureShaderBGR, self).__init__()

        self.vs = \
        """
            #version 450 
            uniform mat3 transform;
            uniform vec2 properties;

            layout (location = 0) in vec2 position;
            layout (location = 1) in vec2 texCoordIn;
            out vec2 texCoordOut;
            out float alpha;
            void main()
            {
                alpha = properties[1];
                vec3 tmp = transform*vec3(position,1.0);
                gl_Position = vec4(tmp.x,tmp.y,0.0,tmp.z);
                texCoordOut = texCoordIn;
            }
        """

        self.fs = \
        """
            #version 450 
            uniform sampler2D text;
            in float alpha;

            in vec2 texCoordOut;
            out vec4 fragColour;

            void main()
            {
                vec3 tmp = vec3(texture(text, texCoordOut));

                fragColour = vec4(tmp, alpha);
            }
        """

    def generate_uniform_functions(self):

        self.transform_uniform_location  = glGetUniformLocation(self.shader_program, "transform")
        self.properties_uniform_location = glGetUniformLocation(self.shader_program, "properties")

        self.uniform_functions["transform"] = lambda M: glUniformMatrix3fv(self.transform_uniform_location, 1, GL_TRUE, M)
        self.uniform_functions["properties"] = lambda P: glUniform2fv(self.properties_uniform_location, 1, P)
  

        return self

class TextureShaderR(ShaderProgram):

    def __init__(self):
        super(TextureShaderR, self).__init__()

        self.vs = \
        """
            #version 450 
            uniform mat3 transform;
            uniform float depth;

            layout (location = 0) in vec2 position;
            layout (location = 1) in vec2 texCoordIn;
            out vec2 texCoordOut;
            void main()
            {
                vec3 tmp = transform*vec3(position,1.0);
                gl_Position = vec4(tmp.x,tmp.y,0.0,tmp.z);
                texCoordOut = texCoordIn;
            }
        """

        self.fs = \
        """
            #version 450 
            uniform sampler2D text;
            uniform vec4 colour;

            in vec2 texCoordOut;
            out vec4 fragColour;

            void main()
            {
                vec4 cl = vec4(texture(text, texCoordOut));
                
                fragColour = vec4(vec3(colour), cl[0]*colour[3]);
            }
        """

    def generate_uniform_functions(self):

        self.transform_uniform_location = glGetUniformLocation(self.shader_program, "transform")
        self.colour_uniform_location = glGetUniformLocation(self.shader_program, "colour")
        self.depth_uniform_location  = glGetUniformLocation(self.shader_program, "depth")

        self.uniform_functions["transform"] = lambda M: glUniformMatrix3fv(self.transform_uniform_location, 1, GL_TRUE, M)
        self.uniform_functions["colour"]    = lambda C: glUniform4fv(self.colour_uniform_location, 1, C)
        self.uniform_functions["depth"]     = lambda D: glUniform1f(self.depth_uniform_location, D)

        return self

class CircleShader(DefaultShader):

    def __init__(self):
        super(CircleShader, self).__init__()

        self.vs = \
        """
            #version 450 
            uniform mat3 transform;
            uniform vec2 properties;

            layout (location = 0) in vec2 position;
            out vec2 fragPosition;
            out float highlight;

            void main()
            {
                highlight = properties[1];

                fragPosition = position;
                vec3 tmp = transform*vec3(position,1.0);
                gl_Position = vec4(tmp.x,tmp.y,0.0,tmp.z);
            }
        """

        self.fs = \
        """
            #version 450 
            uniform vec4 colour;
            in float highlight;

            in vec2 fragPosition;
            out vec4 fragColour;

            void main()
            {
                float inside = fragPosition[0]*fragPosition[0] + fragPosition[1]*fragPosition[1];
                float alpha0 = 1.0f - smoothstep(1.0 - fwidth(inside), 1.0, inside);
                //float alpha1 = smoothstep(0.8 - fwidth(inside), 0.8, inside);

                fragColour = colour;
                fragColour[0] = clamp(fragColour[0]*highlight, 0.0, 1.0);
                fragColour[1] = clamp(fragColour[1]*highlight, 0.0, 1.0);
                fragColour[2] = clamp(fragColour[2]*highlight, 0.0, 1.0);
                fragColour[3] = alpha0;
            }
        """

    def generate_uniform_functions(self):

        self.transform_uniform_location  = glGetUniformLocation(self.shader_program, "transform")
        self.colour_uniform_location     = glGetUniformLocation(self.shader_program, "colour")
        self.properties_uniform_location = glGetUniformLocation(self.shader_program, "properties")

        self.uniform_functions["transform"] = lambda M: glUniformMatrix3fv(self.transform_uniform_location, 1, GL_TRUE, M)
        self.uniform_functions["colour"]    = lambda C: glUniform4fv(self.colour_uniform_location, 1, C)
        self.uniform_functions["properties"] = lambda P: glUniform2fv(self.properties_uniform_location, 1, P)

        return self

class LoadingShader(ShaderProgram):

    def __init__(self):
        super(LoadingShader, self).__init__()

        self.vs = \
        """
            #version 450 
            uniform mat3 transform;
            uniform float depth;

            layout (location = 0) in vec2 position;
            layout (location = 1) in vec2 texCoordIn;

            out vec2 texCoord;
            out vec2 fragmentPosition;

            void main()
            {
                fragmentPosition = position;
                texCoord = texCoordIn;
                vec3 tmp = transform*vec3(position,1.0);
                gl_Position = vec4(tmp.x,tmp.y,0.0,tmp.z);
            }
        """

        self.fs = \
        """
            #version 450 
            uniform int stage;
            uniform sampler2D textureSampler;
            uniform vec4 animation_details;

            in vec2 fragmentPosition;
            in vec2 texCoord;
            out vec4 fragmentColour;

            // Stage 0

            float t = 2.5f;
            float scale = 0.3f;
            float circleSize = 0.2f; 
            float yoffset = 0.0f; // Centering deppends on top panel size

            float rFrac = 0.11f;

            // Stage 1
            float r1 = 2.5f;

            // Uniform to named variable
            float time      = animation_details[0];
            float aspect    = animation_details[1];
            float timeStage = animation_details[2];
            float duration  = animation_details[3]; 

            vec3 vicosRed = vec3(226.0/255, 61.0/255, 40.0/255.0);

            void main()
            {
                if (stage == 0)
                {
                    float p = ((sin(time*t*0.5f) + 1.0f)/2)*circleSize + scale;
                    float zto = (p - scale)/(1.0f - scale);

                    float r0 = p*0.4f;
                    float r1 = p*0.5f;

                    float fx = fragmentPosition[0]*aspect;
                    float fy = fragmentPosition[1];

                    float d = fx*fx + (fy + yoffset)*(fy + yoffset);
                    bool a = (d >= r0) && (d <= r1);
        
                    fragmentColour = vec4(vicosRed, a ? 1.0f : 0.0f);
                }
                else if ((stage == 1) && (timeStage <= duration))
                {
                    float r0 = (((sin((time - timeStage)*t*0.5f) + 1.0f)/2)*circleSize + scale)*0.6f;
                    
                    float i = sin(3.14159*timeStage/duration*0.5f);
                    float rFinal = r0*(1.0f - i) + r1*i;
                    float rSmall = rFinal*(1.0f - rFrac);

                    float fx = fragmentPosition[0]*aspect;
                    float fy = fragmentPosition[1] + yoffset;

                    bool a1 = (fx*fx + fy*fy <= rFinal*rFinal) && (fx*fx + fy*fy >= rSmall*rSmall);
                    bool a2 = (fx*fx + fy*fy < rSmall*rSmall);

                    if (a1)
                    {
                        fragmentColour = vec4(vicosRed, 1.0f);
                    }
                    else if (a2)
                    {
                        fragmentColour = vec4(vec3(texture(textureSampler, texCoord)), 1.0f);
                    }
                    else
                    {
                        fragmentColour = vec4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                }
                else
                {
                    fragmentColour = vec4(vec3(texture(textureSampler, texCoord)), 1.0f);
                }
            }
        """

    def generate_uniform_functions(self):

        self.transform_uniform_location = glGetUniformLocation(self.shader_program, "transform")
        self.stageUniformLocation   = glGetUniformLocation(self.shader_program, "stage")
        self.depth_uniform_location = glGetUniformLocation(self.shader_program, "depth")
        self.animation_detailsUniformLocation = glGetUniformLocation(self.shader_program, "animation_details")

        self.uniform_functions["transform"] = lambda M: glUniformMatrix3fv(self.transform_uniform_location, 1, GL_TRUE, M)
        self.uniform_functions["stage"]     = lambda S: glUniform1i(self.stageUniformLocation, S)
        self.uniform_functions["depth"]     = lambda D: glUniform1f(self.depth_uniform_location, D)
        self.uniform_functions["animation_details"] = lambda D: glUniform4fv(self.animation_detailsUniformLocation, 1, D)

        return self

class MenuDrawerShader(ShaderProgram):

    def __init__(self):
        super(MenuDrawerShader, self).__init__()

        self.vs = \
        """
            #version 450 
            uniform mat3 transform;

            layout (location = 0) in vec2 position;
            out vec2 fPosition;

            void main()
            {
                fPosition = (position + 1.0f)*0.5f;
                vec3 tmp = transform*vec3(position,1.0);
                gl_Position = vec4(tmp.x,tmp.y,0.0,tmp.z);
            }
        """

        self.fs = \
        """
            #version 450 
            uniform int pressed;

            in vec2 fPosition;
            out vec4 fragmentColour;

            float b = 0.04f;

            void main()
            {
                bool isClick = pressed == 1;

                float x = fPosition[0];
                float y = fPosition[1];

                int L = 3;
                float s = 1.0f/(2.0f*L - 1.0f);

                float a = 0.0f;
                for (int i = 0; i < L; i++)
                {
                    float sTop = float(i)*2.0f*s;
                    float sBot = s + sTop;

                    bool big = y <= 1.0f - sTop && 
                               y >= 1.0f - sBot &&
                               x <= 1.0f &&
                               x >= 0.0f;
                    bool small = y <= 1.0f - sTop - b && 
                                 y >= 1.0f - sBot + b &&
                                 x <= 1.0f - b &&
                                 x >= b;  

                  
                    a += float(!small && big || (isClick && big));
                }

                fragmentColour = vec4(1.0f, 1.0f, 1.0f, a);
            }
        """

    def generate_uniform_functions(self):

        self.transform_uniform_location = glGetUniformLocation(self.shader_program, "transform")
        self.pressed_uniform_location   = glGetUniformLocation(self.shader_program, "pressed")

        self.uniform_functions["transform"] = lambda M: glUniformMatrix3fv(self.transform_uniform_location, 1, GL_TRUE, M)
        self.uniform_functions["pressed"] = lambda A: glUniform1i(self.pressed_uniform_location, A)

        return self
