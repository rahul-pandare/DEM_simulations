import numpy as np
import matplotlib.pyplot as plt
import time
import random
'''
July 3, 2025
RVP

Script to generate random packng of paricles in a 2D box for a  given packing fraction.
This script uses a Verlet integration method to simulate particle dynamics while gradually growing,
ensuring that particles do not overlap.
This script has no wall potential, but uses a contact model to handle particle interactions.
'''

class GrowingPacking:
    def __init__(self, npp, phi, ar, vr, dr=0.1, dt=1e-2, a1=1, kp=1, pm=1):
        self.npp = npp
        self.phi = phi
        self.ar = ar
        self.vr = vr
        self.dr = dr * a1
        self.dt = dt
        self.kp = kp
        self.pm = pm
        self.a1 = a1

        # particle counts
        self.nps = round(npp / ((1/ar**2) * ((1/vr) - 1) + 1))
        self.npl = npp - self.nps

        self.radiiList = self.nps * [a1] + self.npl * [a1 * ar]
        random.shuffle(self.radiiList)

        self.pa = sum(np.pi * r**2 for r in self.radiiList)
        self.lx = np.sqrt(self.pa / phi)
        self.walls = [0, self.lx]

        np.random.seed(42)
        self.pos = np.random.rand(npp, 2) * (self.lx - 2 * np.max(self.radiiList)) + np.max(self.radiiList)
        self.vel = np.random.randn(npp, 2)

        # create radius growth plan
        self.rlinAll = [list(np.arange(self.dr, r + self.dr, self.dr)) for r in self.radiiList]
        max_len = max(len(x) for x in self.rlinAll)
        for x in self.rlinAll:
            while len(x) < max_len:
                x.append(x[-1])
        self.rlinAll = np.array(self.rlinAll)
        self.iter = self.rlinAll.shape[1]
        self.figCount = 1
        self.s = 0

    def verlet_integration(self, acc):
        return self.pos + self.vel * self.dt + 0.5 * acc * self.dt**2

    def apply_pbc(self):
        self.pos = self.pos % self.lx

    def compute_accelerations(self, pr):
        acc = np.zeros_like(self.pos)
        for i in range(self.npp):
            for j in range(i + 1, self.npp):
                dist = self.pos[j] - self.pos[i]
                dist -= self.lx * np.round(dist / self.lx)
                dist_norm = np.linalg.norm(dist)
                rr = pr[i] + pr[j]
                if dist_norm < rr:
                    overlap = rr - dist_norm
                    force = overlap * (dist / dist_norm)
                    acc[i] -= force / self.pm
                    acc[j] += force / self.pm
        return acc

    def particle_contact(self, pr):
        for i in range(self.npp):
            for j in range(i + 1, self.npp):
                dist = self.pos[j] - self.pos[i]
                dist -= self.lx * np.round(dist / self.lx)
                norm_dist = np.linalg.norm(dist)
                rr = pr[i] + pr[j]
                if norm_dist < rr:
                    norm_dir = dist / norm_dist
                    rel_vel = self.vel[j] - self.vel[i]
                    vel_mag = np.dot(rel_vel, norm_dir)
                    if vel_mag < 0:
                        delta = (np.dot(rel_vel, dist) / norm_dist**2) * dist
                        self.vel[i] += delta
                        self.vel[j] -= delta

    def compute_overlaps(self, pr):
        overlaps = []
        for i in range(self.npp):
            for j in range(i + 1, self.npp):
                dist = self.pos[j] - self.pos[i]
                dist -= self.lx * np.round(dist / self.lx)
                norm_dist = np.linalg.norm(dist)
                rr = pr[i] + pr[j]
                if norm_dist < rr:
                    overlaps.append((rr - norm_dist) * 100 / rr)
        return overlaps

    def plot_circles(self, pr, disp=True, saveFig=False):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        shifts = [-self.lx, 0, self.lx]

        for i in range(self.npp):
            color = '#323232' if self.radiiList[i] == np.min(self.radiiList) else '#ADD8E6'
            for sx in shifts:
                for sy in shifts:
                    shifted = self.pos[i] + np.array([sx, sy])
                    circle = plt.Circle(shifted, pr[i], facecolor=color, edgecolor='None')
                    ax.add_artist(circle)

        ax.set_xlim(self.walls)
        ax.set_ylim(self.walls)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color((0, 0, 0, 0.5))

        if saveFig:
            fig.savefig(f'frame_{self.figCount:03d}.png', bbox_inches='tight')
        if disp:
            plt.show()
        plt.close()

    def run(self, verbose=True, save_path=None):
        start_time = time.time()

        for jj in range(self.iter):
            pr = self.rlinAll[:, jj]
            paS = sum(np.pi * r**2 for r in pr)
            phiS = round(paS / self.lx**2, 3)

            if verbose and jj % (self.iter // 10) == 0:
                print(f'{phiS:.3f}, {jj * 100 // self.iter:.0f}%')
                self.dt *= 0.8

            while True:
                self.s += 1
                acc = self.compute_accelerations(pr)
                self.pos = self.verlet_integration(acc)
                self.apply_pbc()
                self.vel += 0.5 * acc * self.dt
                self.particle_contact(pr)

                if self.s % 100 == 0:
                    self.plot_circles(pr)

                overlap = self.compute_overlaps(pr)
                mean_overlap = np.mean(overlap) if overlap else 0
                count_overlap = sum(1 for x in overlap if x != 0)

                if mean_overlap == 0 or (mean_overlap < 0.05 and count_overlap > 0.2 * self.npp):
                    break

        # Save final configuration
        if save_path:
            with open(save_path, 'w') as f:
                f.write('# x y radius\n')
                for i in range(self.npp):
                    f.write(f'{self.pos[i,0]} {self.pos[i,1]} {pr[i]:.1f}\n')

        elapsed_time = (time.time() - start_time) / 60
        print(f'Done\nElapsed time: {elapsed_time:.2f} mins')

# Example usage:
if __name__ == '__main__':
    sim = GrowingPacking(npp=10, phi=0.7, ar=1.4, vr=0.5, dr=0.1)
    sim.run(save_path='finalPacking.txt')