import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from racetrack import RaceTrack
from racecar import RaceCar
from controller import lower_controller, controller

class Simulator:

    def __init__(self, rt : RaceTrack, raceline = None, headless = False, speed = 1):
        self.headless = headless
        self.speed = speed  # Render 1 in every 'speed' frames
        self.frame_counter = 0  # Counter for frame skipping
        
        if not headless:
            matplotlib.rcParams["figure.dpi"] = 200
            matplotlib.rcParams["font.size"] = 8

        self.rt = rt
        self.raceline = raceline  # Store the raceline
        
        if not headless:
            self.figure, self.axis = plt.subplots(1, 1)
            self.axis.set_xlabel("X"); self.axis.set_ylabel("Y")
        else:
            self.figure = None
            self.axis = None

        self.car = RaceCar(self.rt.initial_state.T)

        self.lap_time_elapsed = 0
        self.lap_finished = False
        self.lap_started = False
        self.track_limit_violations = 0
        self.currently_violating = False
        self.simulation_time = 0.0  # Track simulation time independently
        self.lap_start_simulation_time = 0.0  # Track when lap started in simulation time
        self.trajectory = []  # Store trajectory points [x, y, speed]

    def check_track_limits(self):
        car_position = self.car.state[0:2]
        
        min_dist_right = float('inf')
        min_dist_left = float('inf')
        
        for i in range(len(self.rt.right_boundary)):
            dist_right = np.linalg.norm(car_position - self.rt.right_boundary[i])
            dist_left = np.linalg.norm(car_position - self.rt.left_boundary[i])
            
            if dist_right < min_dist_right:
                min_dist_right = dist_right
            if dist_left < min_dist_left:
                min_dist_left = dist_left
        
        centerline_distances = np.linalg.norm(self.rt.centerline - car_position, axis=1)
        closest_idx = np.argmin(centerline_distances)
        
        to_right = self.rt.right_boundary[closest_idx] - self.rt.centerline[closest_idx]
        to_left = self.rt.left_boundary[closest_idx] - self.rt.centerline[closest_idx]
        to_car = car_position - self.rt.centerline[closest_idx]
        
        right_dist = np.linalg.norm(to_right)
        left_dist = np.linalg.norm(to_left)
        
        proj_right = np.dot(to_car, to_right) / right_dist if right_dist > 0 else 0
        proj_left = np.dot(to_car, to_left) / left_dist if left_dist > 0 else 0
        
        is_violating = proj_right > right_dist or proj_left > left_dist
        
        if is_violating and not self.currently_violating:
            self.track_limit_violations += 1
            self.currently_violating = True
        elif not is_violating:
            self.currently_violating = False

    def run(self):
        try:
            if self.lap_finished:
                exit()

            # Always advance simulation by exactly one timestep
            # This ensures consistent simulation time regardless of callback frequency
            desired = controller(self.car.state, self.car.parameters, self.rt)
            cont = lower_controller(self.car.state, desired, self.car.parameters)
            self.car.update(cont)
            
            # Increment simulation time by car's timestep
            self.simulation_time += self.car.time_step
            
            # Update lap time if lap has started (calculate from start time)
            if self.lap_started and not self.lap_finished:
                self.lap_time_elapsed = self.simulation_time - self.lap_start_simulation_time
            
            self.update_status()
            self.check_track_limits()

            # Increment frame counter and check if we should render this frame
            self.frame_counter += 1
            should_render = (self.frame_counter % self.speed == 0)

            if not self.headless and should_render:
                self.figure.canvas.flush_events()
                self.axis.cla()

                self.rt.plot_track(self.axis)

                self.axis.set_xlim(self.car.state[0] - 200, self.car.state[0] + 200)
                self.axis.set_ylim(self.car.state[1] - 200, self.car.state[1] + 200)

            # Always track trajectory (even if not rendering this frame)
            if not self.headless:
                self.trajectory.append([self.car.state[0], self.car.state[1], self.car.state[3]])
            
            # Only render when should_render is True
            if not self.headless and should_render:
                # Plot trajectory
                if len(self.trajectory) > 1:
                    trajectory_array = np.array(self.trajectory)
                    for i in range(len(trajectory_array) - 1):
                        speed = trajectory_array[i, 2]
                        if speed < 20:
                            color = 'red'
                        elif speed < 50:
                            ratio = (speed - 20) / 30
                            color = (1.0, ratio * 0.65, 0.0)
                        else:
                            ratio = (speed - 50) / 50
                            color = ((1.0 - ratio), 0.65 + ratio * 0.35, 0.0)
                        self.axis.plot(trajectory_array[i:i+2, 0], trajectory_array[i:i+2, 1], 
                                     color=color, linewidth=2, alpha=0.8)

                self.axis.arrow(
                    self.car.state[0], self.car.state[1], \
                    self.car.wheelbase*np.cos(self.car.state[4]), \
                    self.car.wheelbase*np.sin(self.car.state[4])
                )

                self.axis.text(
                    self.car.state[0] + 195, self.car.state[1] + 195, "Lap completed: " + str(self.lap_finished),
                    horizontalalignment="right", verticalalignment="top",
                    fontsize=8, color="Red"
                )

                self.axis.text(
                    self.car.state[0] + 195, self.car.state[1] + 170, "Lap time: " + f"{self.lap_time_elapsed:.2f}",
                    horizontalalignment="right", verticalalignment="top",
                    fontsize=8, color="Red"
                )

                self.axis.text(
                    self.car.state[0] + 195, self.car.state[1] + 155, "Track violations: " + str(self.track_limit_violations),
                    horizontalalignment="right", verticalalignment="top",
                    fontsize=8, color="Red"
                )

                self.axis.text(
                    self.car.state[0] + 195, self.car.state[1] + 140, "Speed: " + f"{self.car.state[3]:.1f} m/s",
                    horizontalalignment="right", verticalalignment="top",
                    fontsize=8, color="Red"
                )

                self.figure.canvas.draw()
            return True

        except KeyboardInterrupt:
            exit()

    def update_status(self):
        progress = np.linalg.norm(self.car.state[0:2] - self.rt.centerline[0, 0:2], 2)

        if progress > 10.0 and not self.lap_started:
            self.lap_started = True
            # Record simulation time when lap starts
            self.lap_start_simulation_time = self.simulation_time
            self.lap_time_elapsed = 0.0
    
        if progress <= 10.0 and self.lap_started and not self.lap_finished:
            self.lap_finished = True
            # Calculate final lap time from simulation time
            self.lap_time_elapsed = self.simulation_time - self.lap_start_simulation_time
            
            # Print results to console
            print("\n" + "="*50)
            print("LAP COMPLETED!")
            print("="*50)
            print(f"Lap Time: {self.lap_time_elapsed:.2f} seconds")
            print(f"Track Violations: {self.track_limit_violations}")
            print("="*50 + "\n")
            
            # Save results to file
            try:
                with open("lap_results.txt", "w") as f:
                    f.write("="*50 + "\n")
                    f.write("LAP RESULTS\n")
                    f.write("="*50 + "\n")
                    f.write(f"Lap Time: {self.lap_time_elapsed:.2f} seconds\n")
                    f.write(f"Track Violations: {self.track_limit_violations}\n")
                    f.write("="*50 + "\n")
                print("Results saved to lap_results.txt")
            except Exception as e:
                print(f"Warning: Could not save results to file: {e}")

    def start(self):
        if self.headless:
            # In headless mode, run simulation loop directly without timer
            while not self.lap_finished:
                self.run()
        else:
            # Run the simulation loop every 1 millisecond with GUI
            self.timer = self.figure.canvas.new_timer(interval=1)
            self.timer.add_callback(self.run)
            self.timer.start()
