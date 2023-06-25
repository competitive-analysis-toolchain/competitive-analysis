# Competitive Analysis Toolchain
The following commands guides the user to run the Competitive analysis toolchain inside a virtual environment.
The toolchain is tested on Ubuntu 16.04 using Python 3.6.

## Installation
1. Create a virtual environment (say, `env`) and activate it
- `$ python3 -m venv env`
- `$ source env/bin/activate`

2. Install the required tools and packages
`pip install -r requirements.txt`

## Usage

1. Navigate to the desired use case folder
-  cart pendulum use case: `$ cd competitive_analysis/cart_pendulum`
-  motion planning use case: `$ cd competitive_analysis/motion_planning`

2. Execute the shell script to start the competitive analysis toolchain:
-  cart pendulum use case: `./run_cart.sh`
-  motion planning use case: `./run_motion.sh`

3. By default, the tool runs for 20 retraining iterations. To change the number of iterations, open the respective `.sh` file and search for "# CEGAR loop". Modify the number of iterations as needed.

