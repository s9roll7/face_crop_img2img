import launch

if not launch.is_installed("ipython"):
    launch.run_pip("install ipython", "requirements for face crop img2img")

if not launch.is_installed("seaborn"):
    launch.run_pip("install ""seaborn>=0.11.0""", "requirements for face crop img2img")

