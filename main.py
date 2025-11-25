from sys import argv

import numpy as np

from simulator import RaceTrack, Simulator, plt

if __name__ == "__main__":
    assert len(argv) >= 3
    track_path = argv[1]
    raceline_path = argv[2]
    headless = "--headless" in argv
    
    # Parse speed parameter (third positional arg or --speed=N)
    speed = 1
    if len(argv) >= 4 and argv[3].isdigit():
        speed = int(argv[3])
    else:
        for arg in argv:
            if arg.startswith("--speed="):
                speed = int(arg.split("=")[1])

    # Load track and attach raceline coordinates to the RaceTrack object
    racetrack = RaceTrack(track_path)
    raceline_data = np.loadtxt(raceline_path, comments="#", delimiter=",")
    # First two columns are x, y of the optimal raceline
    racetrack.raceline = raceline_data[:, :2]

    simulator = Simulator(racetrack, headless=headless, speed=speed)
    simulator.start()
    if not headless:
        plt.show()
