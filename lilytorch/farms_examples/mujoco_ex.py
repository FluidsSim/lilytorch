import mujoco
import glfw
import numpy as np

xml = """
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba=".9 .9 .9 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Initialize GLFW
glfw.init()
window = glfw.create_window(1200, 900, "MuJoCo with Rectangle Overlay", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# Create MuJoCo rendering contexts
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

# Set camera
cam.azimuth = 90
cam.elevation = -20
cam.distance = 3
cam.lookat = np.array([0.0, 0.0, 0.5])

while not glfw.window_should_close(window):
    # Physics step
    mujoco.mj_step(model, data)

    # Get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(viewport, scene, context)

    # Draw rectangle overlay - top-left corner, semi-transparent blue
    rect = mujoco.MjrRect(10, viewport_height - 110, 200, 100)  # x, y, width, height
    mujoco.mjr_rectangle(rect, 0.2, 0.3, 0.8, 0.5)  # RGBA color

    # Swap buffers
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()