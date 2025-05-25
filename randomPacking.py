import numpy as np
'''
May 24, 2025
RVP

Script to generate random packng of paricles in a 2D box for a  given packing fraction.
This script uses a Verlet integration method to simulate particle dynamics, ensuring that particles do not overlap.
This script has no wall potential, but uses a contact model to handle particle interactions.
'''
class randomPacking:
    def __init__(self, npp, pr, phi, lx, dt=1e-2, pm=1.0, kp=1.0, kw=1.0):
        self.npp = npp
        self.pr = pr
        self.phi = phi
        self.lx = lx
        self.dt = dt
        self.pm = pm
        self.kp = kp
        self.kw = kw
        
        np.random.seed(42)
        self.pos = (np.random.rand(npp, 2).astype(np.float64) - 0.5) * lx
        self.vel = np.zeros_like(self.pos)

    def verlet_integration(self, acc):
        return self.pos + self.vel * self.dt + 0.5 * acc * self.dt**2

    def compute_accelerations(self):
        acc = np.zeros_like(self.pos)
        for i in range(self.npp):
            for j in range(i + 1, self.npp):
                dist = self.pos[j] - self.pos[i]
                dist_norm = np.linalg.norm(dist)
                rr = self.pr[i] + self.pr[j]
                if dist_norm < rr:
                    overlap = rr - dist_norm
                    norm_dir = dist / dist_norm
                    force = overlap * norm_dir
                    acc[i] -= force / self.pm
                    acc[j] += force / self.pm
        return acc

    def wall_contact(self):
        for i in range(self.npp):
            for dim in range(2):
                if self.pos[i, dim] < self.pr[i] - self.lx / 2:
                    self.vel[i, dim] = abs(self.vel[i, dim]) * self.kw
                elif self.pos[i, dim] > self.lx / 2 - self.pr[i]:
                    self.vel[i, dim] = -abs(self.vel[i, dim]) * self.kw

    def particle_contact(self):
        for i in range(self.npp):
            for j in range(i + 1, self.npp):
                dist = self.pos[j] - self.pos[i]
                norm_dist = np.linalg.norm(dist)
                rr = self.pr[i] + self.pr[j]
                if norm_dist < rr:
                    norm_dir = dist / norm_dist
                    rel_vel = self.vel[j] - self.vel[i]
                    vel_mag = np.dot(rel_vel, norm_dir)
                    if vel_mag < 0:
                        delta = (np.dot(rel_vel, dist) / norm_dist**2) * dist * self.kp
                        self.vel[i] += delta
                        self.vel[j] -= delta

    def compute_overlaps(self):
        overlaps = []
        for i in range(self.npp):
            for j in range(i + 1, self.npp):
                rr = self.pr[i] + self.pr[j]
                dist = self.pos[j] - self.pos[i]
                norm_dist = np.linalg.norm(dist)
                if norm_dist < rr:
                    overlaps.append((rr - norm_dist) * 100 / rr)
        return overlaps

    def run(self, max_frames=1000, max_overlap=1):
        frame_count = 0
        while True:
            frame_count += 1
            
            # simulation steps
            acc       = self.compute_accelerations()
            self.pos  = self.verlet_integration(acc)
            self.vel += 0.5 * acc * self.dt
            self.wall_contact()
            self.particle_contact()
            overlap = self.compute_overlaps()
            
            if np.max(overlap) < max_overlap:
                break
            if frame_count > max_frames:
                print("\nOptimal positions not found, returning best attempt\n")
                break
            
        return self.pos
