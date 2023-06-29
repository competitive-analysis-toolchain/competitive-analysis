# Competitive Analysis Toolchain
The following commands guides the user to run the Competitive analysis toolchain inside a virtual environment.
The toolchain is tested on Ubuntu 16.04 using Python 3.6.10 and on Ubuntu 22.04 using Python 3.10.6.

## Pre-requisites
1. Java: The HyST tool requires Java. Download and install Java 1.8+ by following the instructions available at [JavaDownloads](https://www.java.com/en/download/help/download_options.html)
2. Flow* Compilation: The Flow* tool needs to be compiled to generate the Flow* binary. Follow these steps to install the required libraries and compile Flow*:
- `$ sudo apt install libgmp3-dev`
- `$ sudo apt install libmpfr-dev libmpfr-doc libmpfr4 libmpfr4-dbg`
- `$ sudo apt install gsl-bin libgsl0-dev`
- `$ sudo apt install bison`
- `$ sudo apt install flex`
- `$ sudo apt install gnuplot-x11`
- `$ sudo apt install libglpk-dev`
- `$ sudo apt install libyaml-cpp-dev`
- The output from Flow* is parsed by our Flowpipeanalyser, which requires the Jsoncpp library. We have used _libjsoncpp-dev 1.7.2-1_ which can be downloaded from [Jsoncpp Releases](https://github.com/open-source-parsers/jsoncpp/releases/tag/1.7.2) or from [Launchpad](https://launchpad.net/ubuntu/+source/libjsoncpp) for Ubuntu. For user's convenience, the required Jsoncpp library files (`/flowstar/lib`), header files (`flowstar/jsoncpp/json`), and make files (`/flowstar/cmake/jsoncpp`) are included in the _flowstar_ folder.
- From the _flowstar_ folder, run the make command: `$ cd flowstar; make`

## Installation
1. Create a virtual environment (say, `env`) and activate it.
- `$ python3 -m venv env`
- `$ source env/bin/activate`

2. Install the required Python packages. Depending on your Python version and Ubuntu distribution, use the appropriate `requirements_ubuntu16_04.txt` or `requirements_ubuntu22_04.txt` file as a reference. These files contain the specific packages needed for the DNN training code.
- For Python 3.6.10 on Ubuntu 16.04: `$ pip install -r requirements_ubuntu16_04.txt`
- For Python 3.10.6 on Ubuntu 22.04: `$ pip install -r requirements_ubuntu22_04.txt`

## Usage

1. Navigate to the desired use case folder
-  For e.g., for cart pendulum use case: `$ cd "$HOME/competitive-analysis/cart_pendulum"`
-  For e.g., for motion planning use case: `$ cd "$HOME/competitive-analysis/motion_planning"`
Note that the subsequent shell script execution assumes this folder structure. Alternatively, you can modify the paths in the shell script for convenience.

2. Execute the shell script to start the competitive analysis toolchain:
-  cart pendulum use case: `./run_cart.sh`
-  motion planning use case: `./run_motion.sh`

3. By default, the tool runs for 20 retraining iterations. To change the number of iterations, open the respective `.sh` file and search for "# CEGAR loop". Modify the number of iterations as needed.

