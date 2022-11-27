# Image Pyramids & Pyramid Blending
![Output example](/Figure_1.png)<br> 

## Overview

This project deals with image pyramids, low-pass and band-pass filtering, and their application in image
blending. In this project I constructed Gaussian and Laplacian pyramids, used these to implement
pyramid blending, and finally compared the blending results when using different filters in the various
expand and reduce operations.
 
I developed it during the course [Image Processing - 67829](https://moodle2.cs.huji.ac.il/nu17/course/view.php?id=67829), taught at the Hebrew University by Prof. Shmuel Peleg during spring semester of 2017/18.
This code was my submission for [exercise #3](ex3.pdf) .

## Setup
Clone the repository, then create a local [virtual environment](https://www.geeksforgeeks.org/python-virtual-environment/#:~:text=A%20virtual%20environment%20is%20a,of%20the%20Python%20developers%20use.):
```bash
$ virtualenv -p /usr/bin/python3 venv
```

Activate the virtual environment with:
```bash
$ source venv/bin/activate
(venv)$
```

Install ```requirements.txt``` using pip:
```bash
(venv)$ pip install -r requirements.txt
```

## Usage
By default, run it with:
```bash
(venv)$ python test3.py
```
To explore different demos, choose your desired function in [test3.py](test3.py) by uncommenting the relevant lines of code inside ```main()```, and then run it again as shown above.