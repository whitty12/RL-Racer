# RL-Worm

## Dependencies

Open AI Gym
	git clone https://github.com/openai/gym.git
    cd gym
    pip install -e .
    
SWIG
    http://www.swig.org
    
PyBox2D
    https://github.com/pybox2d/pybox2d

## Learner Requirements
step function: gets the next action based on the current state
    parameters:
        a numpy array of shape VP_H, VP_W, 3 representing the state
    returns
        the next action to take as a list [x, y, z]
            where x is the steering
            y is the gas
            z is the break
update function: updates the learner given reward information
    parameters:
        a numpy array of shape VP_H, VP_W, 3 representing the state
        an integer reward
        a boolean flag done
    returns nothing

To Use:
    in racer.py, in main, change the learner initialization to your learner
    Note: you may need to import the file containing your learner

