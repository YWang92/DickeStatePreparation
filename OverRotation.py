import qutip
import numpy as np
import random
import time
import scipy.special

from datetime import datetime



def ensemble_initial(N):
    # initialize the spin ensemble as a single large spin, assuming no single-spin errors.
    #initial spin coherent state of the spin ensemble, i.e., |+>^{\otimes N}
    vacuum = qutip.basis(N+1, 0)
    y_halfpi = (-1.j * np.pi/2 * qutip.spin_Jy(N/2)).expm()
    ensemble = y_halfpi * vacuum  

    return ensemble/ensemble.norm()



def ancilla_projection(N, ensemble, estimation, rotation_angle, over_rotation):

    rotation = (-1.j * rotation_angle * (qutip.spin_Jz(N/2) - estimation) ).expm()* ( -1.j * over_rotation * qutip.spin_Jz(N/2) ).expm()
    # The controlled-rotation with over-rotation, which is induced by inaccurate control of ancilla qubit control.

    p0 = 1/2 + 1/4 * (   ensemble.dag()* (rotation + rotation.dag() ) * ensemble )
    p1 = 1/2 - 1/4 * (   ensemble.dag()* (rotation + rotation.dag()) * ensemble )
    p0, p1 = abs(p0[0][0][0]), abs(p1[0][0][0])
    # p0 and p1 are the probabilities of obtaining 0 and 1 measurement outcome
    
    if np.random.rand() < p0:
        ensemble = ( qutip.qeye(N+1) + 1 * rotation ) * ensemble / np.sqrt(p0)
        outcome = 0
    else:
        ensemble = ( qutip.qeye(N+1) - 1 * rotation ) * ensemble / np.sqrt(p1)
        outcome = 1

    ensemble = ensemble/ensemble.norm()
    
    return ensemble, outcome


def parity_attempt(estimation, pe_round, ensemble, N, over_sigma):
    # The evolution time for the controlled global rotation
    # pe_round starts at 0
    rotation_angle = np.pi/2**pe_round

    over_rotation = np.random.normal(0, over_sigma)

    ensemble, outcome = ancilla_projection(N, ensemble, estimation, rotation_angle, over_rotation)

    return ensemble, outcome



def preparation_attempt(N, over_sigma, attempts):
    log = int( np.ceil( np.log2(N)/2 ) )
    # The preparation is actually a phase estimation circuit

    ensemble = ensemble_initial(N)
    estimation = 0

    for pe_round in range(log+1):
        n0, n1 = 0, 0
        
        while abs(n0+n1) < attempts:   
            ensemble, outcome = parity_attempt(estimation, pe_round, ensemble, N, over_sigma = over_sigma)

            if outcome == 0:
                n0 += 1
            elif outcome ==1:
                n1 += 1

        # Performing majority voting
        if n0 > n1:
            outcome = 0
        else:
            outcome = 1

        estimation += 2**(pe_round) * outcome

    # Find the smallest possible estimation (with respect to absolute value).
    if estimation > 2**(pe_round + 1) / 2:
        estimation = int(estimation - 2**(pe_round+1) + N/2)
    else:
        estimation = int(estimation + N/2)

    
    return ensemble, estimation



def preparation_experiment(experiment_rounds, over_sigma, attempts, N):
    Fidelity_eachround = []

    for _ in range(experiment_rounds):
        ensemble, estimation = preparation_attempt(N, over_sigma, attempts)

        Fidelity_eachround.append( abs(ensemble[N - estimation][0][0])**2 )
        # estimation is  m_z + N/2, which is the number of spins in |0>

    return Fidelity_eachround




import pickle
def experiment_running(N, over_sigma, experiment_rounds, Attempts):
    file = open("InaccurateControl.txt", "w") 
    file.write("The simulation with over_sigma = {}, starts at: {} \n'".format(over_sigma, datetime.now()) )
    file.close()

    fidelity, Std_fidelity = [], []

    for attempts in Attempts:
        file = open("InaccurateControl.txt", "a") 
        file.write("attempts: {}, starting at: {} \n'".format(attempts, datetime.now()) )
        file.write('-----------------------------------------------------------------------------\n')
        file.close()


        Fidelity_eachround = preparation_experiment(experiment_rounds, over_sigma, attempts, N)
        
        average_fidelity = np.sum(Fidelity_eachround)/experiment_rounds
        std_fidelity = np.sqrt( np.var(Fidelity_eachround)  )

    
        file = open("InaccurateControl.txt", "a") 
        file.write("average_fidelity: {}, std fidelity: {} \n".format( average_fidelity, std_fidelity))
        file.write('-----------------------------------------------------------------------------\n')
        file.write('-----------------------------------------------------------------------------\n')
        file.close()

        fidelity.append( average_fidelity )
        Std_fidelity.append( std_fidelity)
        
    return fidelity, Std_fidelity


