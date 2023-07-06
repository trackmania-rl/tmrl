import subprocess
import time

# Find the window ID for the window with name "Lutris"
print("find lutris window ...")
proc = subprocess.Popen(['xdotool', 'search', '--onlyvisible', '--name', 'Lutris'], stdout=subprocess.PIPE)
w_lutris = proc.stdout.readline().decode().strip()
print(f"\tlutris window detected, id={w_lutris}")

print("find ubisoft connect in lutris ...")
subprocess.run(['xdotool', 'windowsize', w_lutris, '640', '640'])
subprocess.run(['xdotool', 'windowmove', w_lutris, '0', '0'])
time.sleep(1)

subprocess.run(['xdotool', 'mousemove', '25', '100'])
subprocess.run(['xdotool', 'click', '1'])
time.sleep(1)

subprocess.run(['xdotool', 'mousemove', '300', '25'])
subprocess.run(['xdotool', 'click', '1'])
time.sleep(1)

subprocess.run(['xdotool', 'key', 'U'])
subprocess.run(['xdotool', 'key', 'b'])
subprocess.run(['xdotool', 'key', 'i'])
time.sleep(1)

subprocess.run(['xdotool', 'mousemove', '300', '100'])
subprocess.run(['xdotool', 'click', '1'])
time.sleep(1)

print("start ubisoft connect ...", end=" ")
subprocess.run(['xdotool', 'mousemove', '250', '600'])
subprocess.run(['xdotool', 'click', '1'])
for i in range(90):
    time.sleep(1)
    if i % 10 == 0:
        print(f"{i}, ")
print("\tubisoft connect should have opened up by now")

# Find the window ID for the window with name "ubisoft connect"
proc = subprocess.Popen(['xdotool', 'search', '--onlyvisible', '--name', 'ubisoft connect'], stdout=subprocess.PIPE)
w_ubisoft = proc.stdout.readline().decode().strip()
print(f"\tubisoft window detected, id={w_ubisoft}")

quit()

print("enter games ...")
subprocess.run(['xdotool', 'mousemove', '200', '50'])
subprocess.run(['xdotool', 'click', '1'])
time.sleep(3)

print("find trackmania ...")
subprocess.run(['xdotool', 'mousemove', '200', '300'])
subprocess.run(['xdotool', 'click', '1'])
time.sleep(1)

print("start trackmania ...")
subprocess.run(['xdotool', 'mousemove', '200', '400'])
subprocess.run(['xdotool', 'click', '1'])
time.sleep(1)
