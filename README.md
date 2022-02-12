# Jetson Nano SLAM Robot

This project contains a collection of (slow) algorithms for odometry, trajectory estimation and mapping from 2D Lidar or stereo cameras.

## Next Steps:

* Code optimizations to approach real-time performance on the Jetson Nano (Re-write everything in C++)
    * Currently using [CuPy](https://cupy.dev/) for performing matrix ops on the GPU, but this is limited by the GPU memory of the device
* More accurate algorithms
    * Deep stereo depth estimation would be a good start for visual SLAM
    * Get a better kinematic model for the robot, and use kalman filters for more accurate localization
* Autonomous exploration
    * Reinforcement learning in sim for learning the control policy
    * Implementation of path planning algorithms (like [A*](https://en.wikipedia.org/wiki/A*_search_algorithm))
