import subprocess
import sys
import time

while True:
    try:
        # 运行你的脚本
        subprocess.check_call(['python', '48objects_fullset_collecting_with_llama3.1_8b.py'])
    except subprocess.CalledProcessError:
        # 如果脚本中断退出，则打印错误信息并重新启动脚本
        print("脚本中断退出，正在重新启动...")
    except KeyboardInterrupt:
        # 如果用户通过键盘中断脚本，则退出循环
        print("用户中断了脚本")
        break
    