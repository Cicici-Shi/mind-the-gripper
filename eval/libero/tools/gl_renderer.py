"""Print the EGL GL_RENDERER string through the same EGL path robosuite uses (MUJOCO_GL=egl)."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from robosuite.renderers.context.egl_context import EGLGLContext
c = EGLGLContext(max_width=64, max_height=64, device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")))
c.make_current()
from OpenGL import GL
print("GL_RENDERER", GL.glGetString(GL.GL_RENDERER).decode())
