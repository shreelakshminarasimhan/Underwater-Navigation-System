# Underwater-Navigation-System

## SNC Navigation Simulator
A real-time Python simulation of the Navigation subsystem (with sensing and communication elements) designed for a deep-sea research submarine operating in the Clarion Clipperton Zone (CCZ). It visualises how Inertial Navigation System (JNS), Doppler Velocity Log, and Acoustic Doppler Velocity Profiler (ADCP) navigation error accumulates over a multi-day survey mission, and how a GPS body corrects that drift.

https://github.com/user-attachments/assets/ad896943-7ab8-4b80-9cce-27cd5feb5d1f

### Why This Exists
Submarines can't use GPS underwater, so their estimated position slowly drifts away from their true position. XXXXXXX *placeholder for more words*

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
*Placeholder for code*
```
git clone
cd snc-simulator
pip install matplotlib
pip install numpy

python snc_main.py
```

The simulator runs entirely on matplotlib and dumpy 

## Deep Sea Survey _ Interactive Game 
An interactive arcade-style game that turns a real submarine navigation-error model into a playable risk-reward mechanic. Built for a live design exhibition, players pilot a submarine on a survey mission and decide when to "spend" a limited number of GPS buoy fixes to correct the navigation drift: balancing survey speed against data quality, exactly as a real mission planner would.



https://github.com/user-attachments/assets/d1413ff5-f918-47b7-8e30-737df8276258


### Why it exists
Navigation drift is an abstract concept and hard to explain to a non-technical audience in a few seconds at an exhibition stand. This game makes it tangible: every second you delay a GPS fix, your submarine's estimated position gets less accurate, and your survey data quality quietly degrades. However GPS fixes cost money and time and so you have to be strategic about when to deploy them (using control systems). Built as a companion piece to a full submarine SNC subsystem simulator, it was demoed live to academic and industry visitors. 

### Features
* Gamepad Support: PS4 controller inout via pyjama, blended with keyboard controls (if controller unavailble)
* Live scoring system tied to the real navigation-drift model — the longer you go without a GPS fix, the lower your data-quality tier (High/ Medium/ Low)

### How to play
1. Steer your submarine around the survey area using the keyboard or gamepad
2. Watch the navigation error grow the longer you go between GPS fixes
3. Deploy a buoy fix (limited number available) to reset your error. But each fix costs time
4. Complete the survey with the best possible data-quality score before time runs out

### Getting Started 
*Placeholder for code*
```
git clone
cd deep-sea-survey
pip install flask pygame

python snc_manual.py
```

Once running, use a connected PS4 controller/ joystick/ keyboard to control the submarine directly. 

### Author 
Shree Lakshminarasimhan - Built as a part of a group submarine design project 

