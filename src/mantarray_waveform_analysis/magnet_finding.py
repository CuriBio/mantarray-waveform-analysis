# -*- coding: utf-8 -*-
"""More accurate estimation of magnet positions."""
import numpy as np
from scipy.optimize import least_squares
import scipy.signal as signal
from numba import njit


@njit()
def meas_field(
    xpos,  # TODO: what is the shape of this array?: <Answer>
    zpos,  # TODO: what is the shape of this array?: <Answer>
    theta,  # TODO: what is the shape of this array?: <Answer>
    ypos,  # TODO: what is the shape of this array?: <Answer>
    phi,  # TODO: what is the shape of this array?: <Answer>
    remn,  # TODO: what is the shape of this array?: <Answer>
    arrays  # TODO: what does this argument signify and what is its shape (if any)>: <Answer>
):
    """Simulate fields using a magnetic dipole model."""

    # TODO: what are these values? What do they signify, what are their units, etc.: <Answer>
    triad = np.asarray([[-2.15, 1.7, 0], [2.15, 1.7, 0], [0, -2.743, 0]])  # sensor locations

    # TODO: what is this value (manta) being calculated?: <Answer>
    # TODO: what is this block of code doing?: <Answer>
    manta = triad + arrays[0] // 6 * np.asarray([0, -19.5, 0]) + (arrays[0] % 6) * np.asarray([19.5, 0, 0])
    for array in range(1, len(arrays)): # Compute sensor positions -- probably not necessary to do each call  # TODO: why?: <Answer>
        manta = np.append(manta, (triad + arrays[array] // 6 * np.asarray([0, -19.5, 0]) + (arrays[array] % 6) * np.asarray([19.5, 0, 0])), axis=0)
    # TODO: What is np.asarray([0, -19.5, 0])?: <Answer>
    # TODO: What is np.asarray([19.5, 0, 0]))?: <Answer>

    fields = np.zeros((len(manta), 3))
    theta = theta / 360 * 2 * np.pi
    phi = phi / 360 * 2 * np.pi

    # TODO: what is this block of code doing?: <Answer>
    for magnet in range(0, len(arrays)):
        unnamed_const_1 = np.pi * (.75 / 2.0) ** 2  # TODO: what is the significance of this constant?: <Answer>
        m = unnamed_const_1 * remn[magnet] * np.asarray(
            [
                np.sin(theta[magnet]) * np.cos(phi[magnet]),
                np.sin(theta[magnet]) * np.sin(phi[magnet]),
                np.cos(theta[magnet])
            ]
        )  # moment vectors
        r = -np.asarray([xpos[magnet], ypos[magnet], zpos[magnet]]) + manta  # radii to moment
        # TODO: what does rAbs signify?: <Answer>
        rAbs = np.sqrt(np.sum(r ** 2, axis=1))

        # simulate fields at sensors using dipole model for each magnet
        for field in range(0, len(r)):
            fields[field] += 3 * r[field] * np.dot(m, r[field]) / rAbs[field] ** 5 - m / rAbs[field] ** 3

    # TODO: why is it necessary to reshape and then divide these values?: <Answer>
    unnamed_const_2 = 4 / np.pi  # TODO: what is the significance of this constant?: <Answer>
    return fields.reshape((1, 3 * len(r)))[0] / unnamed_const_2


def objective_function_ls(pos, Bmeas, arrays):
    """Cost function."""
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


def processData(dataName):
    """Process data from instrument."""
    fileName = f'{dataName}.txt'
    numWells = 24
    numSensors = 3
    numAxes = 3
    memsicCenterOffset = 2 ** 15
    memsicMSB = 2 ** 16
    memsicFullScale = 16
    gauss2MilliTesla = .1

    # TODO: what does this value signify?: <Answer>
    config = np.loadtxt(fileName, max_rows=1, delimiter=', ').reshape((numWells, numSensors, numAxes))

    # TODO: what is this block of code doing?: <Answer>
    activeSensors = np.any(config, axis=1)
    spacerCounter = 1
    timestampSpacer = [0]
    dataSpacer = []
    for (wellNum, sensorNum), status in np.ndenumerate(activeSensors):
        if status:
            numActiveAxes = np.count_nonzero(config[wellNum, sensorNum])
            for numAxis in range(1, numActiveAxes + 1):
                dataSpacer.append(timestampSpacer[spacerCounter - 1] + numAxis)
            timestampSpacer.append(timestampSpacer[spacerCounter - 1] + numActiveAxes + 1)
            spacerCounter += 1

    # TODO: is this converting the timestamp values into seconds?: <Answer>
    timestamps = np.loadtxt(fileName, skiprows=1, delimiter=', ', usecols=tuple(timestampSpacer[:-1])) / 1000000
    # TODO what is the format of the incoming data?: <Answer>
    data = (
        np.loadtxt(fileName, skiprows=1, delimiter=', ', usecols=tuple(dataSpacer)) - memsicCenterOffset
    ) * memsicFullScale / memsicMSB * gauss2MilliTesla

    numSamples = timestamps.shape[0] - 2
    fullData = np.zeros((numWells, numSensors, numAxes, numSamples))
    fullTimestamps = np.zeros((numWells, numSensors, numSamples))

    # TODO: what is this block of code doing?: <Answer>
    dataCounter = 0
    for (wellNum, sensorNum, axisNum), status in np.ndenumerate(config):
        if status:
            fullData[wellNum, sensorNum, axisNum] = data[2:, dataCounter]
            dataCounter += 1

    # TODO: what is this block of code doing?: <Answer>
    timestampCounter = 0
    for (wellNum, sensorNum), status in np.ndenumerate(activeSensors):
        if status:
            fullTimestamps[wellNum, sensorNum] = timestamps[2:, timestampCounter]
            timestampCounter += 1

    # TODO: what is this block of code doing?: <Answer>
    outliers = np.argwhere(fullData < -.3)
    for outlier in outliers:
        fullData[outlier[0], outlier[1], outlier[2], outlier[3]] = np.mean(
            [
                fullData[outlier[0], outlier[1], outlier[2], outlier[3] - 1],
                fullData[outlier[0], outlier[1], outlier[2], outlier[3] + 1],
            ]
        )

    return fullData


def getPositions(data):
    """Generate initial guess data and run least squares optimizer on instrument data to get magnet positions."""
    numSensors = 3
    numAxes = 3

    xpos_est = []
    ypos_est = []
    zpos_est = []
    theta_est = []
    phi_est = []
    remn_est = []

    # TODO: why is passing this dummy data in first necessary?: <Answer>
    dummy = np.asarray([1])
    meas_field(dummy, dummy, dummy, dummy, dummy, dummy, dummy)

    # TODO: what is this block of code doing? What is being stored in arrays?: <Answer>
    arrays = []
    for array in range(0, data.shape[0]):
        if data[array, 0, 0, 0]:
            arrays.append(array)

    # TODO: is this guess just hardcoded? What does each value mean?: <Answer>
    guess = [0, -5, 95, 1, 0, -575] #x, z, theta, y, phi remn
    x0 = []  # TODO what is x0?
    # TODO: what is this block of code doing? What do i and j signify?: <Answer>
    for i in range(0, 6):
        for j in arrays:
            if i == 3:
                x0.append(guess[i] - 19.5 * (j // 6))
            elif i == 0:
                x0.append(guess[i] + 19.5 * (j % 6))
            else:
                x0.append(guess[i])

    res = []
    # TODO: what is this block of code doing? What are the significance of 500 and 150 here?: <Answer>
    for i in range(0, 500):  # 150
        if len(res) > 0:
            x0 = np.asarray(res.x)

        # TODO: is it necessary for this variabe to exist? It's always just 1: <Answer>
        increment = 1

        # TODO: what is this block of code doing?: <Answer>
        Bmeas = np.zeros(len(arrays) * 9)
        for j in range(0, len(arrays)): # divide by how many columns active, should be divisible by 4
            Bmeas[j*9:(j+1) * 9] = np.asarray(data[arrays[j], :, :, increment * i].reshape((1, numSensors * numAxes))) # Col 5

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

    # TODO: do we need to use all the estimations to run analysis in the SDK?: <Answer>
    return [np.asarray(xpos_est),
           np.asarray(ypos_est),
           np.asarray(zpos_est),
           np.asarray(theta_est),
           np.asarray(phi_est),
           np.asarray(remn_est)]


def find_magnet_position(fields, baseline):
    # TODO: which of these (1 or 2) is the correct way to process the incoming data?: <Answer>
    # (1) # TODO: what is this method doing?: <Answer>
    mtFields = processData("Durability_Test_11162021_data_90min")[:, :, :, 1500:2000] - np.tile(processData("Durability_Test_11162021_data_Baseline")[:, :, :, 1500:1502], 250)
    # (2) # TODO: what is this method doing?: <Answer>
    mtFields2 = processData("2021-10-28_16-45-57_data_3rd round second with plate")[:, :, :, 1500:2000]  # TODO: why does the data need to be sliced like this [:, :, :, 1500:2000]?: <Answer>
    meanOffset = np.mean(processData("2021-10-28_16-44-21_data_3rd round second run baseline")[:, :, :, 1500:2000], axis=3)
    for i in range (0, len(mtFields2)):
        mtFields2[:, :, :, i] = mtFields2[:, :, :, i] - meanOffset

    outputs = getPositions(mtFields)

    # TODO: Is filtering here necessary? Can filtering be done before passing data to getPositions?: <Answer>
    high_cut = 30 # Hz
    b, a = signal.butter(4, high_cut, 'low', fs=100)
    outputs = signal.filtfilt(b, a, outputs, axis=1)