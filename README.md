# DEM Simulations

<table>
<tr>
<td align="center">
  <strong>Particle Interactions</strong><br/>
  <img src="randomPacking_growing.mp4" alt="random packing GIF" width="350"/><br/>
  <em>Initializing system with random particle positions by growing.</em>
</td>
<td align="center">
  <strong>Particle Velocities</strong><br/>
  <img src="NP_100_phi_0.5_Hertzian-Mindlin.gif" alt="particle trajectory and particle-particle contacts" width="410"/><br/>
  <em>Particle trajectories and particle-particle contacts (Hertzian-Mindlin).</em>
</td>
</tr>
</table>

<!-- <p align="center">
  <strong>Simulation Movie</strong><br/>
  <img src="NP_100_phi_0.5_Hertzian-Mindlin.gif"
       alt="particle trajectory and particle-particle contacts"
       width="400"/><br/>
  <em>Particle trajectories and particle-particle contacts (Hertzian-Mindlin)</em>
</p>
<br/> -->

**Analysis of Discrete Element Method (DEM) simulations with two contact models: Cundall-Strack and Hertz-Mindlin**

This repository contains code and data for DEM simulations comparing two widely used contact models:

### 📁 Repository Contents

1. **DEM Simulation Code**
   - Python implementations of DEM simulations using:
     - **Cundall-Strack model**
     - **Hertz-Mindlin model**

2. **Initial Configuration**
   - Random packing generator to initialize particles inside a simulation box.

3. **Simulation Data**
   - Interaction data files for both simulation cases (`.txt` format).

4. **Simulation Results**
   - GIF animations showing the simulation results.

---

### 📖 Documentation & Resources

- **Walkthrough Blog Post**  
  Step-by-step guide to building a DEM simulation:  
  [Building a DEM Simulation](https://medium.com/@pandare.rahul/building-a-dem-simulation-59389d196d97)

- **Model Explanation & Comparison**  
  Detailed explanation of the Cundall-Strack and Hertz-Mindlin models, including their key differences:  
  [DEM Simulation Report (PDF)](https://rahul-pandare.github.io/files/blogs/DEM_simulations_report.pdf)

---

### 📌 Notes

- The simulation is implemented in **Python** and is intended for educational and research purposes.
- The report highlights differences in how each model handles normal and tangential contact forces, stiffness, and damping effects.

---

Feel free to fork the repository, run simulations, and explore!
