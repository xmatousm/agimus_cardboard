# 2020-10-23

import numpy as np
from geometry.basic import sqc

I33 = np.eye( 3 )
O3 = np.zeros( (3,1) )

def EutoRt( E, u1, u2 ):

    D = np.array( [ [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0] ] )

    u,_,vt = np.linalg.svd( E )

    u = u * np.linalg.det(u); # now det( u ) > 0
    vt = vt * np.linalg.det(vt); # now det( v ) > 0

    t = u[:,2:];

    Ra = u @ D @ vt
    Rb = u @ D.T @ vt

    #  two possible rotations, four possible cameras -> use cheirality to select
    # the correct one

    P1 = np.hstack( ( I33, O3 ) )
    Pa = np.hstack( ( Ra, t ) )
    # Pb = ( Ra, -t )
    # Pc = ( Rb, t )
    # Pd = ( Rb, -t )

    # reprojection
    X = Pu2X_dlt( P1, Pa,u1, u2 )

    # cheirality test to determine correct configuration (Ra or Rb, +- t )
    c1 = X[2] * X[3]
    c2 = ( Pa @ X )[2] * X[3]
    
    if ( ( c1 > 0 ) * ( c2 > 0 ) ).any(): # Pa is correct
        if ( c1 > 0 ).all() and ( c2 > 0 ).all(): # % ALL points
            return Ra, t
  
    elif ( ( c1 < 0 ) * ( c2 < 0 ) ).any(): # Pb is correct
        if ( c1 < 0 ).all() and ( c2 < 0 ).all(): # % ALL points
            return Ra, -t

    else: # all( c1.*c2 ) <= 0
        Ht = np.vstack( ( P1, np.hstack( ( -2 * vt[2], -1 ) ) ) )
        c3 = X[2] * ( Ht @ X )[3]
 
        if ( c3 > 0 ).all(): # Pc is correct, ALL points
            return Rb, t

        if ( c3 < 0 ).all(): # Pd is correct, ALL points
            return Rb, -t

    return None, None


def Pu2X_dlt( P1, P2, u1, u2 ):
    l = u1.shape[1]

    X = np.zeros( ( 4, l ) )

    for i in range( 0, l ):
        A = np.vstack( ( u1[0,i] * P1[1] - u1[1,i] * P1[0],
                         u1[0,i] * P1[2] - u1[2,i] * P1[0],
                         u1[1,i] * P1[2] - u1[2,i] * P1[1], 
                         u2[0,i] * P2[1] - u2[1,i] * P2[0], 
                         u2[0,i] * P2[2] - u2[2,i] * P2[0], 
                         u2[1,i] * P2[2] - u2[2,i] * P2[1] ) )

        _, _, vh = np.linalg.svd( A )
        X[:,i] = vh[-1].T

    return X

def err_F_sampson( F, u1, u2 ):
    u1 = u1 / u1[-1]
    u2 = u2 / u2[-1]
    
    Fu1 = F @ u1
    Fu2 = F.T @ u2
    
    return ( ( u2 * Fu1 ).sum( axis=0 )**2 / 
             ( Fu1[0]**2 + Fu1[1]**2 + Fu2[0]**2 + Fu2[1]**2 ) )

def err_P_reproj( P, X, u ):
    ux = P @ X
    return ( ( ux / ux[-1] - u / u[-1] )**2 ).sum( axis=0 )
    

def u_correct_sampson_e( F, u1, u2 ):
    u1 = u1 / u1[-1]
    u2 = u2 / u2[-1]
    n = u1.shape[1]
    
    Fu1 = F @ u1
    Fu2 = F.T @ u2

    coeff = ( ( u2 * Fu1 ).sum( axis=0 ) / 
              ( Fu1[0]**2 + Fu1[1]**2 + Fu2[0]**2 + Fu2[1]**2 ) )

    Jvec = np.vstack( ( Fu2[[0,1]], Fu1[[0,1]] ) )

    X = np.vstack( ( u1[[0,1]], u2[[0,1]] ) )
    newX = X - coeff * Jvec

    nu1 = np.vstack( ( newX[[0,1]], np.ones( n ) ) )
    nu2 = np.vstack( ( newX[[2,3]], np.ones( n ) ) )
    
    return nu1, nu2


def P2F( P1, P2 ):

     # The first camera center.
    C1 = np.vstack( ( - np.linalg.inv( P1[:,:3] ) @ P1[:,[3]], 1 ) )
    e2 = P2 @ C1 # epipole in the second camera

    P1p = P1.T @ np.linalg.inv( P1 @ P1.T ) # Pseudoinverse of the first camera
    F = sqc( e2 ) @  P2 @ P1p

    return F


def elemH( *args ):
    if len( args ) % 2 != 0:
        raise Exception( 'Even-sized list of param-value pairs expected' ) # TODO

    H = np.eye( 3 )

    for i in range( 0, len( args ), 2 ):
        key = args[ i ];
        val = args[ i + 1 ]

        if key == 'rot1' or key == 'rotx':
            ca = np.cos( val );  sa = np.sin( val )
            H = np.array( [[1, 0, 0], [0, ca, -sa], [0, sa, ca]] ) @ H

        elif key == 'rot2' or key == 'roty':
            ca = np.cos( val );  sa = np.sin( val )
            H = np.array( [[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]] ) @ H

        elif key == 'rot3' or key == 'rotz':
            ca = np.cos( val );  sa = np.sin( val )
            H = np.array( [[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]] ) @ H

        elif key == 'du' or key == 'dx':
            H = np.array( [[1, 0, val], [0, 1, 0], [0, 0, 1]] ) @ H

        elif key == 'dv' or key == 'dy':
            H = np.array( [[1, 0, 0], [0, 1, val], [0, 0, 1]] ) @ H

        elif key == 's':
            H = np.array( [[val, 0, 0], [0, val, 0], [0, 0, 1]] ) @ H

        elif key == 'su' or key == 'sx':
            H = np.array( [[val, 0, 0], [0, 1, 0], [0, 0, 1]] ) @ H

        elif key == 'sv' or key == 'sy':
            H = np.array( [[1, 0, 0], [0, val, 0], [0, 0, 1]] ) @ H

        elif key == 'q':
            H = np.array( [[1, val, 0], [0, 1, 0], [0, 0, 1]] ) @ H

        else:
            raise Exception( 'Unknown param ' + key ) # TODO xc

    return H
