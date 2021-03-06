# encoding: UTF-8
##
##程序说明：
##作者：赵伟     学号：        联系方式：
##
import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat

pi = 3.14
steering = 0  # initial steering
velocity = 4.0  # the velocity of the vehicle
wheelbase = 8  # the length of wheelbase
dt = 0.05  # time interval
iWp = 1  # index to first waypoint
wp = np.array([[0, -20, -60, -20, 0, 20, 60, 20, 0],
              [-40, 0, 20, 40, 80, 40, 20, 0, -40]])
iPos = 1
rob = np.array([[0, -wheelbase, -wheelbase], [0, -4, 4]])


# % For details, please refer to Section 3.2.1 of R. Smith, M. Self,
# % "Estimating uncertain spatial relationships in robotics,"
# % Autonomous Robot Vehicles, pp. 167-193, 1990.
# %   Xwa - should be of size 3*1 or 1*3
# %   Xab - can be 3*n or 2*n matrix
# % Seeded by Tim Bailey
# % Modified by Samuel on 7 May 2013
def compound(Xwa, Xab):
    rot = np.array([np.cos(Xwa[2]), -np.sin(Xwa[2]), np.sin(Xwa[2]), np.cos(Xwa[2])])
    Xwb = rot * Xab[0:1, :] + repmat(Xwa[0:1], 1, np.size(Xab[1]))

    # if Xwb is a pose and not a point
    if np.size(Xab[0]) == 2:
        Xwb = piTopi(Xwa[2, :] + Xab[2])
    return Xwb


# % Input: array of angles.
# % Output: normalised angles.
# % Tim Bailey 2000, modified 2005.
# % Note: either rem or mod will work correctly
# % angle = mod(angle, 2 * pi);
# % mod() is very slow for some versions of MatLab ( not a builtin function)
# % angle = rem(angle, 2 * pi); % rem() is typically builtin
def piTopi(angle):
    twopi = 2 * pi
    angle -= twopi * np.round(angle / twopi)  # np.ceil对数进行取整
    for i in range(len(angle)):
        if angle[i] >= pi:
            angle[i] -= twopi
        else:
            angle[i] += twopi
    return angle


def compSteer(x, wp, iwp, G, dt):
    maxG = 60 * pi / 80  # radians, maximum steering angle(-MAXG < g < MAXG)
    rateG = 20 * pi / 180  # rad / s, maximum rate of change in steer angle
    minD = 0.5
    cwp = wp[:, iwp-1]
    d2 = (cwp[0] - x[0])**2 + (cwp[1] - x[1])**2
    if d2 < (minD**2):
        iwp += 1  # switch to next
        if iwp > np.size(wp):
            iwp = 0
        cwp = wp[:, iwp-1]  # next waypoint

    # compute change in G to point towards current waypoint
    deltaG = piTopi(np.arctan2(cwp[1] - x[1], cwp[0] - x[0]) - x[2] - G)
    # limit steering rate
    maxDelta = rateG * dt
    if abs(deltaG) > maxDelta:
        deltaG = np.sign(deltaG) * maxDelta
    # limit sterr angle
    G += deltaG
    if abs(G) > maxG:
        G = np.sign(G) * maxG
    return G, iwp, d2


if __name__ == '__main__':
    # Initialize figure
    fig = plt.figure()
    plt.ion()  # continuously plot
    pos = np.zeros([3, 1])
    path = np.zeros([3, 3390])
    idx =0
    while iWp != 0:
        plt.cla()
        pathPlot = plt.plot(0, 0, 'r.', lw=10)
        plt.plot(wp[0, :], wp[1, :], 'g*', lw=2)
        plt.plot(wp[0, :], wp[1, :], 'c:', lw=2)
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        plt.axis([-110, 100, -110, 100])
        idx+=1
        steering, iWp, d2 = compSteer(pos, wp, iWp, steering, dt)  # compute the steering and next waypoint
        pos[0] += velocity * dt * np.cos(steering + pos[2, :])
        pos[1] += velocity * dt * np.sin(steering + pos[2, :])
        pos[2] += velocity * dt * np.sin(steering) / wheelbase
        pos[2] = piTopi(pos[2])
        robPos = compound(pos, rob)
        path[:, iPos+1] = pos[:,0]
        iPos += 1
        pathPlot = plt.plot(path[0, 0:iPos], path[1, 0:iPos], 'r-', lw=1)
        plt.draw()
        plt.pause(0.0001)
        if idx%10==0:
            print(steering)
            print(iWp)
            print(d2)

    print('-dpng', 'path.png')
    plt.show()
