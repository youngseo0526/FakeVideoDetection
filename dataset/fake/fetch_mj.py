import glob
import shutil

for f in (set(glob.glob("mj_all/*.jpg")) - set(glob.glob("mj_all/*thumb*.jpg"))):
    shutil.move(f, "mj_big")
