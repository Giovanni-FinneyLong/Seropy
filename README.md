All visuals & data are hosted: [Here](https://www.dropbox.com/sh/s136nj2b780e22d/AACy854x31kk4U11daFO1Z-0a?dl=0)

Python 3.3 or later required. ImageMagick must be installed to generate gifs.

An (in progress) writeup of the project is hosted: [Here](https://drive.google.com/open?id=0B5YXBruzm8zDUWxkbkQ4TXdEWUU)

# Installation
1. Clone repository: 'git clone git@bitbucket.org:gfinneylong/serotonin.git'
2. Install required packages: python setup.py install
3. Copy config.py into myconfig.py
4. Fill in the 'Folders' section of myconfig.py
5. Configure the 'Switches' section of myconfig.py
6. Run sero.py : python sero.py

Note: Sometimes the below images fail to load, largely due to their size. If this is the case, right click the 'IMAGE' icon and open in a new tab to save a copy

### A Top Down View of the c57b16 dataset, colored by recursive_depth ###
![IMAGE](https://www.dropbox.com/s/4smgqwqimxn4e0s/c57b16_top_depth.png?dl=1)

### A Top Down View of the c57b16 dataset, colored by 3d blob###
![IMAGE](https://www.dropbox.com/s/z4is7p762le1ia2/c57b16_blob3d.png?dl=1)

### A Top Down View of the swellshark dataset, colored by recursive_depth ###
![IMAGE](https://www.dropbox.com/s/s5kom6kc162javc/swell_depth_top.png?dl=1)

### A Top Down View of the swellshark dataset, colored by 3d blob###
![IMAGE](https://www.dropbox.com/s/odqyi3lgg48noqn/swell_blob3d_top.png?dl=1)

### An example of the stitching algorithm, which is used to construct blob3ds from layers of blob2ds (outlined)
![3D-GIF](https://www.dropbox.com/s/a471w8z70jwav7n/Test_Example_of_Point_Matching.gif?dl=1)

### TODOs / Upcoming: ###
* Add blob3d lines to plot in serodraw.py
* Document visualization methods
* Create test suite
* Do more runtime exception checking / assertions
* Add better visuals to readme (not from perpendicular)
* Bug - Some (rare) blob3ds have children id's that have been removed (from being combined with other blob3ds). These specific child ids were meant to have been removed. If this has occured, it will crash execution when visualizing. The temporary solution is to load after running the first time, which will bypass this issue.
* Complete setup.py & pkg-info, confirm that setup.py does complete installation
* Add tags once above is complete, upload tags to pypi, allowing installation with 'pip install Seropy'