# -*- coding: utf-8 -*-
"""More accurate estimation of magnet positions."""
from typing import Any

import numpy as np
from scipy.optimize import least_squares
import scipy.signal as signal
from numba import njit
from nptyping import NDArray

# Kevin (12/1/21): Sensor locations relative to origin
SENSOR_DISTANCES_FROM_CENTER_POINT = np.asarray([[-2.15, 1.7, 0], [2.15, 1.7, 0], [0, -2.743, 0]])

WELL_VERTICAL_SPACING = np.asarray([0, -19.5, 0])  # TODO Tanner (12/2/21): make 19.5 a constant
WELL_HORIZONTAL_SPACING = np.asarray([19.5, 0, 0])
WELLS_PER_ROW = 6

FULL_CIRCLE_DEGREES = 360
FULL_CIRCLE_RADIANS = 2 * np.pi
TODO_CONST_1 = FULL_CIRCLE_RADIANS / FULL_CIRCLE_DEGREES

# Kevin (12/1/21): This is part of the dipole model
TODO_CONST_2 = 4 * np.pi

# Kevin (12/1/21): This is the volume of the magnet for computing its dipole moment
TODO_CONST_3 = np.pi * (.75 / 2.0) ** 2


@njit()
def meas_field(
    xpos: NDArray[(1, Any), float],
    zpos: NDArray[(1, Any), float],
    theta: NDArray[(1, Any), float],
    ypos: NDArray[(1, Any), float],
    phi: NDArray[(1, Any), float],
    remn: NDArray[(1, Any), float],
    arrays: NDArray[(1, Any), float], # TODO rename this var if it doesn't get removed
):
    """Simulate fields using a magnetic dipole model."""
    triad = SENSOR_DISTANCES_FROM_CENTER_POINT.copy()

    # Kevin (12/1/21): Manta gives the locations of all active sensors on all arrays with respect to a common point
    # Computing the locations of each centrally located point about which each array is to be distributed,
    # for the purpose of offsetting the values in triad by the correct well spacing
    manta = triad + arrays[0] // WELLS_PER_ROW * WELL_VERTICAL_SPACING + (arrays[0] % WELLS_PER_ROW) * WELL_HORIZONTAL_SPACING
    for array in range(1, len(arrays)): # Compute sensor positions under each well -- probably not necessary to do each call  # TODO: why?: The values in "triad" and "manta" relate to layout of the board itself -- they don't change at all so long as the board doesn't
        manta = np.append(manta, triad + arrays[array] // WELLS_PER_ROW * WELL_VERTICAL_SPACING + (arrays[array] % WELLS_PER_ROW) * WELL_HORIZONTAL_SPACING, axis=0)

    fields = np.zeros((len(manta), 3))
    # Tanner (12/2/21): numba doesn't like using *= here, for some reason it thinks the dtype of phi and theta are int64 yet they are float64
    theta = theta * TODO_CONST_1  # magnet pitch
    phi = phi * TODO_CONST_1  # magnet yaw

    # Kevin (12/1/21): This simulates the fields for each magnet at each sensor on the device.
    # Each magnet has particular values for xpos -> remn. 
    # These are iteratively optimized to match the simulated fields within a certain tolerance, at which point the algorithm terminates
    for magnet in range(0, len(arrays)):
        # Kevin (12/1/21): compute moment vectors based on magnet strength and orientation
        moment_vectors = TODO_CONST_3 * remn[magnet] * np.asarray(
            [
                np.sin(theta[magnet]) * np.cos(phi[magnet]),
                np.sin(theta[magnet]) * np.sin(phi[magnet]),
                np.cos(theta[magnet])
            ]
        )
        # Kevin (12/1/21): compute distance vector from origin to moment
        r = -np.asarray([xpos[magnet], ypos[magnet], zpos[magnet]]) + manta
        # Kevin (12/1/21): Calculate the euclidian distance from the magnet to a given sensor
        rAbs = np.sqrt(np.sum(r ** 2, axis=1))
        # Kevin (12/1/21): simulate fields at sensors using dipole model for each magnet
        for field in range(0, len(r)):
            # TODO Tanner (12/2/21): If it is necessary to speed the algorithm up, this is the best place to attempt an optimization as the algorithm spends a significant amount of time performing this calculation
            fields[field] += 3 * r[field] * np.dot(moment_vectors, r[field]) / rAbs[field] ** 5 - moment_vectors / rAbs[field] ** 3
    # Kevin (12/1/21): Reshaping to match the format of the data coming off the mantarray
    return fields.reshape((1, 3 * len(r)))[0] / TODO_CONST_2


def objective_function_ls(pos, Bmeas, arrays):
    """Cost function to be minimized by the least squares."""
    # x, z, theta y, phi, remn
    pos = pos.reshape(6, len(arrays))
    x = pos[0]
    z = pos[1]
    theta = pos[2]
    y = pos[3]
    phi = pos[4]
    remn = pos[5]

    Bcalc = np.asarray(meas_field(x, z, theta, y, phi, remn, arrays))

    return Bcalc - Bmeas


def get_positions(data):
    """Generate initial guess data and run least squares optimizer on instrument data to get magnet positions.
    
    Takes an array indexed as [well, sensor, axis, timepoint]
    Data should be the difference of the data with plate on the instrument and empty plate calibration data
    Assumes 3 active sensors for each well, that all active wells have magnets, and that all magnets have the well beneath them active
    """
    numSensors = 3
    numAxes = 3

    # TODO Tanner (12/2/21): The final length of these should be known, so could try to init them as arrays
    xpos_est = []
    ypos_est = []
    zpos_est = []
    theta_est = []  # angle about y
    phi_est = []  # angle about z
    remn_est = []  # remanence

    # Kevin (12/1/21): run meas_field with some dummy values so numba compiles it. There needs to be some delay before it's called again for it to compile
    dummy = np.asarray([1])
    meas_field(dummy, dummy, dummy, dummy, dummy, dummy, dummy)

    # Kevin (12/1/21): This determines which wells are active  # Tanner (12/2/21): We may be able to remove this value throughout the code since all wells/sensors/axes will are always active as of now
    arrays = []
    for array in range(0, data.shape[0]):
        if data[array, 0, 0, 0]:
            arrays.append(array)

    # Kevin (12/1/21): This value is dependent on where the plate sits relative to the sensors
    initial_pos_guess = [0, -5, 95, 1, 0, -575]  # x, z, theta, y, phi remn
    # Kevin (12/1/21): Each magnet has its own positional coordinates and other characteristics depending on where it's located in the consumable. Every magnet
    # position is referenced with respect to the center of the array beneath well A1, so the positions need to be adjusted to account for that, e.g. the magnet in
    # A2 has the x/y coordinate (19.5, 0), so guess is processed in the below loop to produce that value. x0 contains the guesses for each magnet at each position
    x0 = []
    for i in range(0, WELLS_PER_ROW):
        for j in arrays:  # TODO Tanner (12/2/21): rename j
            # TODO Tanner (12/2/21): figure out what is significant about 0 and 3
            if i == 3:
                x0.append(initial_pos_guess[i] - 19.5 * (j // WELLS_PER_ROW))
            elif i == 0:
                x0.append(initial_pos_guess[i] + 19.5 * (j % WELLS_PER_ROW))
            else:
                x0.append(initial_pos_guess[i])

    # Kevin (12/1/21): Run the algorithm on each time index. The algorithm uses its previous outputs as its initial guess for all datapoints but the first one
    for i in range(0, data.shape[-1]):
        print("###", i)

        # Kevin (12/1/21): This sorts the data from processData into something that the algorithm can operate on; it shouldn't be necessary if you combine this method and processData
        Bmeas = np.zeros(len(arrays) * 9)
        for j in range(0, len(arrays)):
            # Kevin (12/1/21): rearrange sensor readings as a 1d vector
            Bmeas[j * 9 : (j + 1) * 9] = np.asarray(data[arrays[j], :, :, i].reshape((1, numSensors * numAxes)))

        res = least_squares(
            objective_function_ls,
            x0,
            args=(Bmeas, arrays),
            method='trf',
            ftol=1e-2
        )

        outputs = np.asarray(res.x).reshape(6, len(arrays))
        xpos_est.append(outputs[0])
        ypos_est.append(outputs[3])
        zpos_est.append(outputs[1])
        theta_est.append(outputs[2])
        phi_est.append(outputs[4])
        remn_est.append(outputs[5])

        # Tanner (12/2/21): set the start point for next loop to the result of this loop
        x0 = np.asarray(res.x)

    # Kevin (12/1/21): I've gotten some strange results from downsampling; I'm not sure why that is necessarily, could be aliasing,
    # could be that the guesses for successive runs need to be really close together to get good accuracy.
    # For large files, you may be able to use the 1D approximation after running the algorithm once or twice "priming"
    return [np.asarray(xpos_est),
           np.asarray(ypos_est),
           np.asarray(zpos_est),
           np.asarray(theta_est),
           np.asarray(phi_est),
           np.asarray(remn_est)]


def find_magnet_position(fields, baseline):
    outputs = get_positions(fields - baseline)

    # TODO: Is filtering here necessary? Can filtering be done before passing data to getPositions?: <Answer>
    high_cut = 30 # Hz
    b, a = signal.butter(4, high_cut, 'low', fs=100)
    outputs = signal.filtfilt(b, a, outputs, axis=1)


##########################################################################################################################
###### KEVIN  ############################################################################################################
##########################################################################################################################


# @njit()
# def meas_field(xpos, zpos, theta, ypos, phi, remn, arrays):

#     triad = np.asarray([[-2.15, 1.7, 0], [2.15, 1.7, 0], [0, -2.743, 0]])  # sensor locations

#     manta = triad + arrays[0] // 6 * np.asarray([0, -19.5, 0]) + (arrays[0] % 6) * np.asarray([19.5, 0, 0])

#     for array in range(1, len(arrays)): #Compute sensor positions -- probably not necessary to do each call
#         manta = np.append(manta, (triad + arrays[array] // 6 * np.asarray([0, -19.5, 0]) + (arrays[array] % 6) * np.asarray([19.5, 0, 0])), axis=0)

#     fields = np.zeros((len(manta), 3))
#     theta = theta / 360 * 2 * np.pi
#     phi = phi / 360 * 2 * np.pi

#     for magnet in range(0, len(arrays)):
#         m = np.pi * (.75 / 2.0) ** 2 * remn[magnet] * np.asarray([np.sin(theta[magnet]) * np.cos(phi[magnet]),
#                                                                   np.sin(theta[magnet]) * np.sin(phi[magnet]),
#                                                                   np.cos(theta[magnet])])  # moment vectors
#         r = -np.asarray([xpos[magnet], ypos[magnet], zpos[magnet]]) + manta  # radii to moment
#         rAbs = np.sqrt(np.sum(r ** 2, axis=1))

#         # simulate fields at sensors using dipole model for each magnet
#         for field in range(0, len(r)):
#             fields[field] += 3 * r[field] * np.dot(m, r[field]) / rAbs[field] ** 5 - m / rAbs[field] ** 3

#     return fields.reshape((1, 3*len(r)))[0] / 4 / np.pi

# # Cost function
# def objective_function_ls(pos, Bmeas, arrays):
#     # x, z, theta y, phi, remn
#     pos = pos.reshape(6, len(arrays))
#     x = pos[0]
#     z = pos[1]
#     theta = pos[2]
#     y = pos[3]
#     phi = pos[4]
#     remn = pos[5]

#     Bcalc = np.asarray(meas_field(x, z, theta, y, phi, remn, arrays))

#     return Bcalc - Bmeas


# # Process data from instrument
# def processData(dataName):
#     fileName = f'{dataName}.txt'
#     numWells = 24
#     numSensors = 3
#     numAxes = 3
#     memsicCenterOffset = 2 ** 15
#     memsicMSB = 2 ** 16
#     memsicFullScale = 16
#     gauss2MilliTesla = .1

#     config = np.loadtxt(fileName, max_rows=1, delimiter=', ').reshape((numWells, numSensors, numAxes))

#     activeSensors = np.any(config, axis=1)
#     spacerCounter = 1
#     timestampSpacer = [0]
#     dataSpacer = []
#     for (wellNum, sensorNum), status in np.ndenumerate(activeSensors):
#         if status:
#             numActiveAxes = np.count_nonzero(config[wellNum, sensorNum])
#             for numAxis in range(1, numActiveAxes + 1):
#                 dataSpacer.append(timestampSpacer[spacerCounter - 1] + numAxis)
#             timestampSpacer.append(timestampSpacer[spacerCounter - 1] + numActiveAxes + 1)
#             spacerCounter += 1

#     timestamps = np.loadtxt(fileName, skiprows=1, delimiter=', ', usecols=tuple(timestampSpacer[:-1])) / 1000000
#     data = (np.loadtxt(fileName, skiprows=1, delimiter=', ',
#                        usecols=tuple(dataSpacer)) - memsicCenterOffset) * memsicFullScale / memsicMSB * gauss2MilliTesla
#     numSamples = timestamps.shape[0] - 2
#     fullData = np.zeros((numWells, numSensors, numAxes, numSamples))
#     fullTimestamps = np.zeros((numWells, numSensors, numSamples))

#     dataCounter = 0
#     for (wellNum, sensorNum, axisNum), status in np.ndenumerate(config):
#         if status:
#             fullData[wellNum, sensorNum, axisNum] = data[2:, dataCounter]
#             dataCounter += 1

#     timestampCounter = 0
#     for (wellNum, sensorNum), status in np.ndenumerate(activeSensors):
#         if status:
#             fullTimestamps[wellNum, sensorNum] = timestamps[2:, timestampCounter]
#             timestampCounter += 1

#     outliers = np.argwhere(fullData < -.3)

#     for outlier in outliers:
#         fullData[outlier[0], outlier[1], outlier[2], outlier[3]] =\
#             np.mean([fullData[outlier[0], outlier[1], outlier[2], outlier[3] - 1],
#                      fullData[outlier[0], outlier[1], outlier[2], outlier[3] + 1]])

#     return fullData

# # Generate initial guess data and run least squares optimizer on instrument data to get magnet positions
# def getPositions(data):
#     numSensors = 3
#     numAxes = 3

#     xpos_est = []
#     ypos_est = []
#     zpos_est = []
#     theta_est = []
#     phi_est = []
#     remn_est = []

#     dummy = np.asarray([1])
#     meas_field(dummy, dummy, dummy, dummy, dummy, dummy, dummy)

#     arrays = []
#     for array in range(0, data.shape[0]):
#         if data[array, 0, 0, 0]:
#             arrays.append(array)
#     print(arrays)

#     guess = [0, -5, 95, 1, 0, -575] #x, z, theta, y, phi remn
#     x0 = []
#     for i in range(0, 6):
#         for j in arrays:
#             if i == 3:
#                 x0.append(guess[i] - 19.5 * (j // 6))
#             elif i == 0:
#                 x0.append(guess[i] + 19.5 * (j % 6))
#             else:
#                 x0.append(guess[i])
#     print(x0)

#     res = []
#     for i in range(0, 100):  # 150
#         print("###", i)
#         if len(res) > 0:
#             x0 = np.asarray(res.x)

#         increment = 1

#         Bmeas = np.zeros(len(arrays) * 9)

#         for j in range(0, len(arrays)): # divide by how many columns active, should be divisible by 4
#             Bmeas[j*9:(j+1) * 9] = np.asarray(data[arrays[j], :, :, increment * i].reshape((1, numSensors * numAxes))) # Col 5

#         res = least_squares(objective_function_ls, x0, args=(Bmeas, arrays),
#                             method='trf', ftol=1e-2)


#         outputs = np.asarray(res.x).reshape(6, len(arrays))
#         xpos_est.append(outputs[0])
#         ypos_est.append(outputs[3])
#         zpos_est.append(outputs[1])
#         theta_est.append(outputs[2])
#         phi_est.append(outputs[4])
#         remn_est.append(outputs[5])

#     return [np.asarray(xpos_est),
#            np.asarray(ypos_est),
#            np.asarray(zpos_est),
#            np.asarray(theta_est),
#            np.asarray(phi_est),
#            np.asarray(remn_est)]
