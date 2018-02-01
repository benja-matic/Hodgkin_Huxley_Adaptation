# Wang-Buzsaki Neurons With Adaptation
Simulations of Wang Buzsaki neurons with adaptation in the form of an M-current.
Includes simulations of single neurons, and competitive networks.
Code is modularly designed: there's a right-hand-side-function, a solver (either a hand-coded one or an interface to a Scipy solver) and a driver (a function that calculates the RHS and solves for the next time step for all specificed time steps).
I used versions of these codes to do simulations of rivalry with realistic conductance based neurons.
