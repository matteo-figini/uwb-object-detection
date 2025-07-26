"""Beamforming"""

import numpy as np

num_tx = 1

num_rx = 2

num_virtual_rx = num_tx * num_rx

slow_time = 300 # acquisition timesteps

fast_time = 120 # range bins

beam_steering_angles = np.deg2rad(np.linspace(-90, 90, 37))


d = 20.55e-3  # Distance between rx antenna elements
f = 7.875e9    # Center frequency of radar pulse
c = 299792458 # Speed of light


l = c/f #wavelength

k = 2*np.pi/l 

dataCube = np.ones((num_virtual_rx,slow_time,fast_time), dtype=complex) # acquired signal num_rx * acquisitions * range_bins

angleDataCube = np.zeros((slow_time,fast_time,beam_steering_angles.shape[0]), dtype="complex") #beamformed signal



def compute_bf_weights(angle_range, delta, k, rx):

        bf_weights = np.zeros((angle_range.shape[0], rx), dtype="complex")

        for i in range(angle_range.shape[0]):
            for j in range(rx):

                bf_weights[i,j] = np.exp(-1j * k * delta * j * np.sin(angle_range[i]))

        return bf_weights



def beamforming(bf_weights):

    for i in range(slow_time):
        for j in range(fast_time):
            for k in range(beam_steering_angles.shape[0]):

                angleDataCube[i][j][k]=np.dot(dataCube[:,i,j],bf_weights[k])




bf_weights = compute_bf_weights(beam_steering_angles, d, k ,num_virtual_rx)

beamforming(bf_weights)




