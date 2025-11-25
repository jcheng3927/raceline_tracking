from sys import argv

import numpy as np

from simulator import RaceTrack, Simulator, plt

if __name__ == "__main__":
    assert len(argv) == 3
    track_path = argv[1]
    raceline_path = argv[2]

    # Load track and attach raceline coordinates to the RaceTrack object
    racetrack = RaceTrack(track_path)
    raceline_data = np.loadtxt(raceline_path, comments="#", delimiter=",")
    # First two columns are x, y of the optimal raceline
    racetrack.raceline = raceline_data[:, :2]

    simulator = Simulator(racetrack)
    simulator.start()
    plt.show()
