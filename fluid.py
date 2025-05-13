import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from fluid_numerics import *

class Fluid:
    def __init__(self, N, boxsize, gamma, courant_fac, tEnd, tOut, slope_limit=True):
        self.N = N
        self.boxsize = boxsize
        self.gamma = gamma
        self.courant_fac = courant_fac
        self.tEnd = tEnd
        self.tOut = tOut
        self.useSlopeLimiting = slope_limit
        self.plotRealTime = True

        self.dx = boxsize / N
        self.vol = self.dx ** 2
        self.t = 0.0
        self.outputCount = 1

        xlin = np.linspace(0.5 * self.dx, boxsize - 0.5 * self.dx, N)
        ylin = np.linspace(0.5 * self.dx, boxsize - 0.5 * self.dx, N)
        self.Y, self.X = np.meshgrid(ylin, xlin)

        self.rho, self.vx, self.vy, self.P = self.init_conditions()
        self.Mass, self.Momx, self.Momy, self.Energy = self.getConserved()

        self.cmap = LinearSegmentedColormap.from_list("custom_rgb", [(0.0, 0.0, 0.0), (0.5, 1.0, 0.0)])
        self.fig = plt.figure(figsize=(4, 4), dpi=80)

    def init_conditions(self):

        w0 = 0.1
        sigma = 0.05 / np.sqrt(2.0)
        density = 1.0 + (np.abs(self.Y - 0.5 * self.boxsize) < 0.005 * self.boxsize)
        vx = -0.5 + (np.abs(self.Y - 0.5 * self.boxsize) < 0.005 * self.boxsize)
        vy = w0 * np.sin(8 * np.pi * self.X) * np.exp(-((self.Y - 0.595 * self.boxsize) ** 2) / (2 * sigma**2))
        P = 2.5 * np.ones_like(self.X)

        return density, vx, vy, P

    def getConserved(self):
        rho, vx, vy, P = self.rho, self.vx, self.vy, self.P
        vol, gamma = self.vol, self.gamma
        Mass = rho * vol
        Momx = rho * vx * vol
        Momy = rho * vy * vol
        Energy = (P / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2)) * vol
        return Mass, Momx, Momy, Energy

    def getPrimitive(self):
        vol, gamma = self.vol, self.gamma
        rho = self.Mass / vol
        vx = self.Momx / rho / vol
        vy = self.Momy / rho / vol
        P = (self.Energy / vol - 0.5 * rho * (vx**2 + vy**2)) * (gamma - 1)
        return rho, vx, vy, P

    def step(self, dt):

        self.rho, self.vx, self.vy, self.P = self.getPrimitive()

        # Compute gradients
        rho_dx, rho_dy = getGradient(self.rho, self.dx)
        vx_dx, vx_dy = getGradient(self.vx, self.dx)
        vy_dx, vy_dy = getGradient(self.vy, self.dx)
        P_dx, P_dy = getGradient(self.P, self.dx)

        if self.useSlopeLimiting:
            rho_dx, rho_dy = slopeLimit(self.rho, self.dx, rho_dx, rho_dy)
            vx_dx, vx_dy = slopeLimit(self.vx, self.dx, vx_dx, vx_dy)
            vy_dx, vy_dy = slopeLimit(self.vy, self.dx, vy_dx, vy_dy)
            P_dx, P_dy = slopeLimit(self.P, self.dx, P_dx, P_dy)

        # Extrapolate half step in time
        rho_p = self.rho - 0.5 * dt * (
            self.vx * rho_dx + self.rho * vx_dx + self.vy * rho_dy + self.rho * vy_dy
        )
        vx_p = self.vx - 0.5 * dt * (self.vx * vx_dx + self.vy * vx_dy + (1 / self.rho) * P_dx)
        vy_p = self.vy - 0.5 * dt * (self.vx * vy_dx + self.vy * vy_dy + (1 / self.rho) * P_dy)
        P_p = self.P - 0.5 * dt * (self.gamma * self.P * (vx_dx + vy_dy) + self.vx * P_dx + self.vy * P_dy)

        # Spatial extrapolation
        rho_XL, rho_XR, rho_YL, rho_YR = extrapolateInSpaceToFace(rho_p, rho_dx, rho_dy, self.dx)
        vx_XL, vx_XR, vx_YL, vx_YR = extrapolateInSpaceToFace(vx_p, vx_dx, vx_dy, self.dx)
        vy_XL, vy_XR, vy_YL, vy_YR = extrapolateInSpaceToFace(vy_p, vy_dx, vy_dy, self.dx)
        P_XL, P_XR, P_YL, P_YR = extrapolateInSpaceToFace(P_p, P_dx, P_dy, self.dx)

        # Fluxes
        fMx, fMMx, fMMy, fE = getFlux(rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, self.gamma)
        fMy, fMMY, fMMX, fEY = getFlux(rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, self.gamma)

        # Apply fluxes
        self.Mass = applyFluxes(self.Mass, fMx, fMy, self.dx, dt)
        self.Momx = applyFluxes(self.Momx, fMMx, fMMX, self.dx, dt)
        self.Momy = applyFluxes(self.Momy, fMMy, fMMY, self.dx, dt)
        self.Energy = applyFluxes(self.Energy, fE, fEY, self.dx, dt)

    def run(self):
        while self.t < self.tEnd:
            self.rho, self.vx, self.vy, self.P = self.getPrimitive()
            dt = self.courant_fac * np.min(
                self.dx / (np.sqrt(self.gamma * self.P / self.rho) + np.sqrt(self.vx**2 + self.vy**2))
            )

            plotThisTurn = False
            if self.t + dt > self.outputCount * self.tOut:
                dt = self.outputCount * self.tOut - self.t
                plotThisTurn = True

            self.step(dt)
            self.t += dt

            if (self.plotRealTime and plotThisTurn) or (self.t >= self.tEnd):
                self.visualize()
                self.outputCount += 1

        plt.savefig("finitevolume.png", dpi=240)
        plt.show()

    def visualize(self, save=True):
        plt.cla()

        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        plt.imshow(self.rho.T, cmap=self.cmap, interpolation='none')
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        self.fig.patch.set_facecolor('none')

        if save:
            frame_filename = f"frames/frame_{self.outputCount:04d}.png"
            plt.savefig(frame_filename, dpi=240, transparent=True, bbox_inches='tight', pad_inches=0)
            print(f"Saved frame {self.outputCount}")

        plt.pause(0.001)