from fluid import Fluid


if __name__ == "__main__":
    fluid = Fluid(N=256, boxsize=2.0, gamma=5/3, courant_fac=0.4, tEnd=4.0, tOut=0.02, slope_limit=False)
    fluid.run()
