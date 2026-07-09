import os
os.environ['MUJOCO_GL'] = 'egl'
import metaworld
import numpy as np
import imageio

print("Initializing ML1 reach-v2...")
try:
    ml1 = metaworld.ML1('reach-v2', seed=0)
    print("ML1 initialized")
    env_cls = list(ml1.train_classes.values())[0]
    print("Env class found")
    env = env_cls(render_mode='rgb_array')
    print("Env created")
    task = ml1.train_tasks[0]
    env.set_task(task)
    env.reset()
    print("Env reset successful")

    print("Rendering...")
    img = env.render()
    print("Render success! Shape:", img.shape)
    imageio.imwrite('test_render.png', img)
    print("Saved test_render.png")
except Exception as e:
    print("An error occurred:")
    import traceback
    traceback.print_exc()
finally:
    if 'env' in locals():
        env.close()
