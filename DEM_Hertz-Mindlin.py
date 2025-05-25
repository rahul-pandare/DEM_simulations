import numpy as np
import matplotlib.pyplot as plt
import random
import time
from randomPacking import randomPacking
import os

'''
May 24, 2025
RVP
Script to simulate particle dynamics using the Hertz-Mindlin contact model.
This script uses a Verlet integration method to simulate rigid-particle dynamics.
'''
savePath  = r"C:\Users\panda\City College Dropbox\Rahul Pandare\CUNY\research\bidisperse_project\miscelleneous\DEM_codes\figures"

start_time = time.time()

# Simulation parameters
a   = 1    # radius of smaller particle (everything scales up with this scale)
npp = 100  # total number of particles
phi = 0.5  # total solid packing fraction

# Particle parameters
pm = 1      # particle mass
mu = 0.7    # Coefficient of friction
E  = 7e10  # Young's modulus in Pa
nu = 0.3    # Poisson's ratio

# Shear modulus (G)
G = E / (2 * (1 + nu))

# Effective Young's modulus (E_eff) for identical particles
E_eff = E / (1 - nu**2)

#E_eff = 2.2e9  # Effective Young's modulus (Pa), e.g., for steel
delta = 0.01    # Overlap
R_eff = 0.5
m_eff = 0.5

# Damping coefficients (tune these as needed)
eta_n = 0.01  # Normal damping coefficient (0 = no damping, 0.1-0.3 typical)
eta_t = 0.05  # Tangential damping coefficient

# Effective stiffness from Hertzian contact
k_eff = 2 * E_eff * np.sqrt(R_eff) * np.sqrt(delta)

# Time step
dt = np.sqrt(m_eff / k_eff)/2

vmax = 5000
 
# List of particle radii in system
pr = npp*[a] + np.random.rand(npp)
random.shuffle(pr)

# Calculating box length
pa = sum(np.pi*(a**2) for a in pr) # area of all particles
lx = np.sqrt(pa/phi)

# Wall limits in x and y both (square box)
walls = [-lx/2, lx/2]

# Random particle positions and velocities
np.random.seed(42)
pos = randomPacking(npp, pr, phi, lx).run()
vel = np.random.randn(npp, 2).astype(np.float64) * vmax

# Initialize tangential displacements (for each possible contact)
tangential_displacements = np.zeros((npp, npp, 2))

def verletIntegration(pos, vel, acc, dt):
    posNew = pos + vel * dt + 0.5 * acc * dt**2
    return posNew

def applyPBC(pos, lx):
    pos = pos % lx  # Wrap positions around the box
    return pos

def computeAccelerations(npp, pos, vel, pr):
    acc = np.zeros_like(pos)
    global tangential_displacements, interaction_data
    
    interaction_data = []

    for i in range(npp):
        for j in range(i + 1, npp):
            distance_vec  = pos[j] - pos[i]
            distance_vec -= lx * np.round(distance_vec / lx)  # PBC correction
            distance_mag = np.linalg.norm(distance_vec)
            rr = pr[i] + pr[j]

            if distance_mag < rr:  # Particle collision
                normal_displacement = rr - distance_mag 
                normal_direction = distance_vec / distance_mag
                reletive_vel  = vel[j] - vel[i]
                
                # --- NORMAL FORCE WITH DAMPING ---
                # Effective radius
                Reff = (pr[i] * pr[j]) / (pr[i] + pr[j])
                
                # Relative normal velocity (scalar)
                velNorm = np.dot(reletive_vel, normal_direction)
                
                # Hertzian + damping (Hunt-Crossley model)
                damping = max(0, -eta_n * velNorm)
                fn_mag = (4/3) * E_eff * np.sqrt(Reff) * min(normal_displacement, delta)**1.5 * (1 + damping)
                fn = fn_mag * normal_direction
                
                # --- TANGENTIAL FORCE WITH DAMPING ---
                tangentDir = np.array([-normal_direction[1], normal_direction[0]])
                velTangent = np.dot(reletive_vel, tangentDir)
                
                # Contact radius (Hertz)
                a = np.sqrt(Reff * normal_displacement)
                
                # Mindlin tangential stiffness
                kt = 8 * G * a
                
                # Update tangential displacement (elastic part)
                prev_disp = tangential_displacements[i, j]
                dxi = velTangent * dt
                dispTangent = prev_disp + dxi
                
                # Trial tangential force (elastic + damping)
                ft_elastic = -kt * dispTangent
                ft_damping = -eta_t * velTangent * tangentDir
                ft_trial = ft_elastic + ft_damping
                
                # Coulomb friction limit
                ft_max = mu * abs(fn_mag)
                ft_mag = np.linalg.norm(ft_trial)
                
                if ft_mag > ft_max:
                    # Sliding: enforce |ft| = μ|fn|
                    ft = -ft_max * (ft_trial / ft_mag)
                    dispTangent = -ft / kt  # Recompute displacement for consistency
                else:
                    # Sticking: use full trial force
                    ft = ft_trial
                
                # Store updated tangential displacement
                tangential_displacements[i, j] = dispTangent
                tangential_displacements[j, i] = -dispTangent
                
                # --- TOTAL FORCE ---
                force = fn + ft
                
                # Update accelerations
                acc[i] -= force / pm
                acc[j] += force / pm
                
                # Log interaction data (optional)
                interaction_data.append([i, j, 
                    pos[i][0], pos[i][1], pos[j][0], pos[j][1],
                    pr[i], pr[j], normal_direction[0], normal_direction[1], tangentDir[0], tangentDir[1],
                    vel[i][0], vel[i][1], vel[j][0], vel[j][1],
                    fn[0], fn[1], ft[0], ft[1], normal_displacement, dispTangent[0], dispTangent[1]])
            
            else:
                # Reset memory if contact is broken
                tangential_displacements[i, j] = np.zeros(2)
                tangential_displacements[j, i] = np.zeros(2)
    
    return acc

def normal_displacements(pos, pr):
    contOverlaps = []
    
    for i in range(npp):
        for j in range(i+1, npp):
            rr       = pr[i] + pr[j] # net center seperation for contacting particles
            distance_vec     = pos[j] - pos[i]
            distance_vec    -= lx * np.round(distance_vec /lx) 
            normDist = np.linalg.norm(distance_vec)
            if normDist < rr:
                contOverlaps.append((rr - normDist)*100/rr) # percentage normal_displacement
    
    return contOverlaps

def plotCircles(pos, pr, disp=False, saveFig=False):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    
    # shifts to show particles on all mirrored sides
    shifts = [-lx, 0, lx]
    
    # Plot each particle and its mirrored versions
    for i in range(npp):
        facecolor = '#8EB6C7'
    
        for shiftX in shifts:
            for shiftY in shifts:
                # Shift the particle position by lx or -lx if it's near the boundary
                shiftedPos = pos[i] + np.array([shiftX, shiftY])
                circle = plt.Circle(shiftedPos, pr[i], facecolor=facecolor, fill=True, edgecolor='None', alpha =0.95)
                ax.add_artist(circle)

    ax.set_xlim(walls)
    ax.set_ylim(walls)
    ax.set_aspect('equal', 'box')
    ax.set_xticks([])
    ax.set_yticks([])
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    if saveFig:
        figFormat = '.png'
        save_folder = os.path.join(savePath, f'NP_{npp}_phi_{phi}_Hertzian-Mindlin')
        os.makedirs(save_folder, exist_ok=True)
        fig.savefig(os.path.join(save_folder, f'frame_{fig_count:03d}{figFormat}'), bbox_inches='tight', dpi=100)
    
    if disp:
        plt.show()
    
    plt.close()

def simulation():
    global pos, vel

    # Step 1: Compute initial acceleration
    acc = computeAccelerations(npp, pos, vel, pr)

    # Step 2: Half-step velocity update
    vel_half = vel + 0.5 * acc * dt

    # Step 3: Update positions
    pos = pos + vel_half * dt
    pos = applyPBC(pos, lx)

    # Step 4: Compute new acceleration with updated velocities
    acc_new = computeAccelerations(npp, pos, vel_half, pr)

    # Step 5: Full-step velocity update
    vel = vel_half + 0.5 * acc_new * dt
    
    # Step 6: Velocity capping (prevent velocity blow-up)
    speeds = np.linalg.norm(vel, axis=1)
    too_fast = speeds > vmax
    if np.any(too_fast):
        vel[too_fast] *= (vmax / speeds[too_fast])[:, np.newaxis]

# Intinializing counters
iteration_count = 0 
fig_count       = 0
t = 0

filename = f'NP_{npp}_phi_{phi}_Hertzian-Mindlin.txt'
with open(filename, 'w') as file:
    file.write('# total_particles, particle_mass, box_length, packing_fraction, fiction coefficient\n')
    file.write(f'{npp} {pm} {lx:.6f} {phi} {mu}\n\n')
    
    file.write('#DEM with Hertz-Mindlin contact model\n')
    file.write('#1  - Particle index i\n')
    file.write('#2  - Particle index j\n')
    file.write('#3-4   - Position of particle i (x, y)\n')
    file.write('#5-6   - Position of particle j (x, y)\n')
    file.write('#7     - Radius of particle i\n')
    file.write('#8     - Radius of particle j\n')
    file.write('#9-10  - Normal vector from i to j\n')
    file.write('#11-12 - Tangential direction vector\n')
    file.write('#13-14 - Velocity of particle i (vx, vy)\n')
    file.write('#15-16 - Velocity of particle j (vx, vy)\n')
    file.write('#17-18 - Normal contact force (fx, fy)\n')
    file.write('#19-20 - Tangential contact force (fx, fy)\n')
    file.write('#21    - Normal displacement (overlap = a_i + a_j - r_ij)\n')
    file.write('#22-23 - Tangential displacement vector (x, y)\n')
    
    while iteration_count < 2e4: #2000 snaps
        iteration_count += 1
        t += dt
        simulation()
        
        if iteration_count%1e1 == 0:
            fig_count += 1
            plotCircles(pos, pr, disp=False, saveFig=True)
            
            #KE = 0.5 * pm * np.sum(np.linalg.norm(vel, axis=1)**2)
            #print(f'KE: {KE:.2e}')
            
            file.write(f'\n\n# time = {t:.3f}\n')
            for entry in interaction_data:
                file.write(' '.join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in entry) + "\n")
            
            max_ov = max(normal_displacements(pos, pr), default=0)
            print(f"Step {iteration_count}, max normal_displacement: {max_ov:.3f}%")
            
print('Done\n')
end_time = time.time()
elapsed_time = (end_time - start_time)/60
print(f"Elapsed time: {elapsed_time:3f} mins")