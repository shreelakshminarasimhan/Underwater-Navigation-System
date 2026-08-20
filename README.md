# Underwater-Navigation-System

## SNC Navigation Simulator
A real-time Python simulation of the Navigation subsystem (with sensing and communication elements) designed for a deep-sea research submarine operating in the Clarion Clipperton Zone (CCZ). It visualises how Inertial Navigation System (JNS), Doppler Velocity Log, and Acoustic Doppler Velocity Profiler (ADCP) navigation error accumulates over a multi-day survey mission, and how a GPS body corrects that drift.

https://github.com/user-attachments/assets/ad896943-7ab8-4b80-9cce-27cd5feb5d1f

### Why This Exists
Submarines can't use GPS underwater, so their estimated position slowly drifts away from their true position. XXXXXXX

### Features:
* Real-time 3D + top-down visualisation of the submarine's true position, estimated position, and accumulated drift, rendered live as the mission runs
* Layered navigation error model (INS baseline, DVL water-tracking aiding, ADCP current correction) based on CEP drift-rate parameters derived from the underlying MATLAB analysis
* 5-state GPS buoy deployment logic: The buoy surfaces, acquires a GPS fix, and corrects the navigation estimate on a realistic timeline (ascent time, acoustic delay, cooldown)
* Waypoint survey path across a geographically accurate CCZ grid (lat/long -> local metric projection)

### How it's built
The simulator is split into independent modules so each piece of the navigation model can be understood, tested, or swapped out on its own:

|Module|Responsibility|
|------|--------------|
|ccz_grid.py|Converts real CCZ latitude and longitude bounds into a flat metric coordinate grid|
|waypoint_path.py|Generates the survey path|
|sub_model.py|Submarine kinematics: true position, heading, and waypoint-following logic|
|drift_model.py|Navigation error accumulation (INS+ DVL water-track + ADCP correction|
|buoy_logic.py|GPS buoy deployment state machine and drift-reset logic|
|snc_main.py|Wires the modules together and drive the live animation|

### Getting Started 
git clone



