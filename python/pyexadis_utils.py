"""@package docstring

ExaDiS python utilities

Implements utility functions for the ExaDiS python binding

* insert_frank_read_src()
* insert_infinite_line()
* insert_prismatic_loop()
* generate_line_config()
* generate_prismatic_config()

* get_segments_end_points()
* get_segments_length()
* dislocation_density()
* dislocation_charge()

* replicate_network()
* combine_networks()
* extract_segments()
* delete_segments()

* read_paradis()
* write_data()
* write_vtk()

Nicolas Bertin
bertin1@llnl.gov
"""

import numpy as np
import pyexadis
from pyexadis_base import NodeConstraints, ExaDisNet
try:
    # Try importing DisNetManager from OpenDiS
    from framework.disnet_manager import DisNetManager
except ImportError:
    # Use dummy DisNetManager if OpenDiS is not available
    from pyexadis_base import DisNetManager

from typing import Tuple


def insert_frank_read_src(cell, nodes, segs, burg, plane, length, center, theta=0.0, linedir=None, numnodes=10):
    """Insert a Frank-Read source into the list of nodes and segments
    cell: network cell
    nodes: list of nodes
    segs: list of segments
    burg: Burgers vector of the source
    plane: habit plane normal of the source
    theta: character angle of the source in degrees
    linedir: line direction of the source
    length: length of the source
    center: center position of the source
    numnodes: number of discretization nodes for the source
    """
    plane = plane / np.linalg.norm(plane)
    if np.abs(np.dot(burg, plane)) >= 1e-5:
        print('Warning: Burgers vector and plane normal are not orthogonal')
    
    if not linedir is None:
        ldir = np.array(linedir)
        ldir = ldir / np.linalg.norm(ldir)
    else:
        b = burg / np.linalg.norm(burg)
        y = np.cross(plane, b)
        y = y / np.linalg.norm(y)
        ldir = np.cos(theta*np.pi/180.0)*b+np.sin(theta*np.pi/180.0)*y
    
    istart = len(nodes)
    for i in range(numnodes):
        p = center -0.5*length*ldir + i*length/(numnodes-1)*ldir
        constraint = NodeConstraints.PINNED_NODE if (i == 0 or i == numnodes-1) else NodeConstraints.UNCONSTRAINED
        nodes.append(np.concatenate((p, [constraint])))
    
    for i in range(numnodes-1):
        segs.append(np.concatenate(([istart+i, istart+i+1], burg, plane)))
    
    return nodes, segs


def insert_infinite_line(cell, nodes, segs, burg, plane, origin, theta=0.0, linedir=None, maxseg=-1, trial=False):
    """Insert an infinite line into the list of nodes and segments
    cell: network cell
    nodes: list of nodes
    segs: list of segments
    burg: Burgers vector of the line
    plane: habit plane normal of the line
    origin: origin position of the line
    theta: character angle of the line in degrees
    linedir: line direction
    maxseg: maximum discretization length of the line
    trial: do a trial insertion only (to test if insertion is possible)
    """
    plane = plane / np.linalg.norm(plane)
    if np.abs(np.dot(burg, plane)) >= 1e-5:
        print('Warning: Burgers vector and plane normal are not orthogonal')
    
    if not linedir is None:
        ldir = np.array(linedir)
        ldir = ldir / np.linalg.norm(ldir)
    else:
        b = burg / np.linalg.norm(burg)
        y = np.cross(plane, b)
        y = y / np.linalg.norm(y)
        ldir = np.cos(theta*np.pi/180.0)*b+np.sin(theta*np.pi/180.0)*y

    h = np.array(cell.h)
    Lmin = np.min(np.linalg.norm(h, axis=0))
    seglength = 0.15*Lmin
    
    if maxseg > 0:
        seglength = np.min([seglength, maxseg])

    length = 0.0
    meet = 0
    maxnodes = 1000
    numnodes = 0
    origin = np.array(origin)
    p = 1.0*origin
    originpbc = 1.0*origin
    while ((~meet) & (numnodes < maxnodes)):
        p += seglength*ldir
        pp = np.asarray(cell.closest_image(Rref=origin, R=p))
        dist = np.linalg.norm(pp-origin)
        if ((numnodes > 0) & (dist < seglength)):
            originpbc = np.asarray(cell.closest_image(Rref=p, R=origin))
            meet = 1
        numnodes += 1

    if numnodes == maxnodes:
        if trial:
            return -1.0
        else:
            print('Warning: infinite line is too long, aborting')
            return nodes, segs

    if trial:
        return np.linalg.norm(originpbc-origin)
    else:
        istart = len(nodes)
        for i in range(numnodes):
            p = origin + 1.0*i/numnodes*(originpbc-origin)
            constraint = NodeConstraints.UNCONSTRAINED
            nodes.append(np.concatenate((p, [constraint])))
        for i in range(numnodes):
            segs.append(np.concatenate(([istart+i, istart+(i+1)%numnodes], burg, plane)))
        return nodes, segs


def insert_prismatic_loop(crystal, cell, nodes, segs, burg, radius, center, maxseg=-1, Rorient=None):
    """Insert a prismatic dislocation loop into the list of nodes and segments
    Input Burgers vector must be of the 1/2<111> type for bcc and 1/2<110> type for fcc.
    Arguments:
    cell: network cell
    nodes: list of nodes
    segs: list of segments
    burg: Burgers vector of the loop
    radius: radius of the loop
    center: center position of the loop
    maxseg: maximum discretization length
    Rorient: crystal orientation matrix
    """ 
    b = -1.0*burg
    
    if crystal in ['BCC', 'bcc']:
        b0 = 1.0/np.sqrt(3.0)*np.array([[1.,1.,1.],[-1.,1.,1.],[1.,-1.,1.],[1.,1.,-1.]])
        bcol = np.abs(np.abs(np.dot(b0, b))-1.0)
        ib = bcol.argmin()
        if bcol[ib] > 1e-5:
            raise ValueError('BCC Burgers vector must be of the 1/2<111> type in insert_prismatic_loop()')
        Nsides = 6
        if 1:
            # Loop with arms on {110} planes (default)
            e = np.array([[-2.0*b[0],b[1],b[2]],[-b[0],-b[1],2.0*b[2]],
                          [b[0],-2.0*b[1],b[2]],[2.0*b[0],-b[1],-b[2]],
                          [b[0],b[1],-2.0*b[2]],[-b[0],2.0*b[1],-b[2]]])
        else:
            # Loop with arms on {112} planes
            e = np.array([[-b[0],0.0,b[2]],[0.0,-b[1],b[2]],
                          [b[0],-b[1],0.0],[b[0],0.0,-b[2]],
                          [0.0,b[1],-b[2]],[-b[0],b[1],0.0]])
        
        n = np.cross(b, e[(np.arange(6)+1)%6]-e[np.arange(6)])
        e = e / np.linalg.norm(e, axis=1)[:,None]
        
    elif crystal in ['FCC', 'fcc']:
        Nsides = 4
        b0 = 1.0/np.sqrt(2.0)*np.array([[0,1,1],[0,-1,1],[1,0,1],[-1,0,1],[1,1,0],[-1,1,0]])
        n01 = np.array([[-1,-1,1],[-1,1,1],[-1,1,1],[1,1,1],[-1,1,1],[1,1,1]])
        n02 = np.array([[1,-1,1],[1,1,1],[-1,-1,1],[1,-1,1],[-1,1,-1],[1,1,-1]])
        bcol = np.abs(np.abs(np.dot(b0, b))-1.0)
        ib = bcol.argmin()
        if bcol[ib] > 1e-5:
            raise ValueError('FCC Burgers vector must be of the 1/2<110> type in insert_prismatic_loop()')
        p1 = n01[ib] / np.linalg.norm(n01[ib])
        p2 = n02[ib] / np.linalg.norm(n02[ib])
        l1 = np.cross(p1, b)
        l1 = l1 / np.linalg.norm(l1)
        l2 = np.cross(p2, b)
        l2 = l2 / np.linalg.norm(l2)
        e = np.array([-0.5*l1-0.5*l2, +0.5*l1-0.5*l2, +0.5*l1+0.5*l2, -0.5*l1+0.5*l2])
        n = np.array([p1, p2, p1, p2])
        
    else:
        raise ValueError('Error: unsupported crystal type = %s in insert_prismatic_loop()' % crystal)
    
    n = n / np.linalg.norm(n, axis=1)[:,None]
    if Rorient is not None:
        Rorient = np.array(Rorient)
        Rorient = Rorient / np.linalg.norm(Rorient, axis=1)[:,None]
        b = np.matmul(b, Rorient.T)
        e = np.matmul(e, Rorient.T)
        n = np.matmul(n, Rorient.T)
    
    istart = len(nodes)
    Nnodes = 0
    for i in range(Nsides):
        l = radius*(e[(i+1)%Nsides]-e[i])
        Nseg = int(np.ceil(np.linalg.norm(l)/maxseg)) if maxseg > 0 else 1
        for j in range(Nseg):
            p = radius*e[i]+1.0*j/Nseg*l+center
            nodes.append(np.concatenate((p, [NodeConstraints.UNCONSTRAINED])))
            n1 = istart+Nnodes
            n2 = istart if (i == Nsides-1 and j == Nseg-1) else n1+1
            segs.append(np.concatenate(([n1, n2], b, n[i])))
            Nnodes += 1
            
    return nodes, segs


def generate_line_config(crystal, Lbox, num_lines, theta=None, maxseg=-1, Rorient=None, seed=-1, verbose=True):
    """Generate a configuration made of straight, infinite dislocation lines
    * Dislocation lines are generated by cycling through the list of signed
      slip systems (+/- Burgers vectors). I.e., for a balanced configuration 
      (neutral Burgers charge), it is advised to use a number of dislocation lines 
      as a multiple of 24 (=12*2), so that dislocation dipoles are created.
    * If a list of character angles (theta) is provided, each dislocation will be
      randomly assigned one of the character angles from the list. If not provided,
      the character angles will be chosen such that the dislocation density is
      roughly equal between all slip systems.
    Arguments:
    * crystal: crystal structure
    * Lbox: box size or cell object
    * num_lines: number of dislocation lines
    * theta: list of possible character angles in degrees
    * maxseg: maximum discretization length of the lines
    * Rorient: crystal orientation matrix
    * seed: seed for random number generation
    * verbose: print information
    """    
    if verbose: print('generate_line_config()')
    
    if crystal in ['BCC', 'bcc']:
        # Define the 12 <111>{110} slip systems
        b = np.array([
            [-1.,1.,1.], [1.,1.,1.], [-1.,-1.,1.], [1.,-1.,1.],
            [-1.,1.,1.], [1.,1.,1.], [-1.,-1.,1.], [1.,-1.,1.],
            [-1.,1.,1.], [1.,1.,1.], [-1.,-1.,1.], [1.,-1.,1.]
        ])
        n = np.array([
            [0.,-1.,1.], [0.,-1.,1.], [0.,1.,1.], [0.,1.,1.],
            [1.,0.,1.], [-1.,0.,1.], [1.,0.,1.], [-1.,0.,1.],
            [1.,1.,0.], [-1.,1.,0.], [-1.,1.,0.], [1.,1.,0.]
        ])
        
    elif crystal in ['FCC', 'fcc']:
        # Define the 12 <110>{111} slip systems
        b = np.array([
            [0.,1.,-1.], [1.,0.,-1.], [1.,-1.,0.],
            [0.,1.,-1.], [1.,0.,1.], [1.,1.,0.],
            [0.,1.,1.], [1.,0.,-1.], [1.,1.,0.],
            [0.,1.,1.], [1.,0.,1.], [1.,-1.,0.]
        ])
        n = np.array([
            [1.,1.,1.], [1.,1.,1.], [1.,1.,1.],
            [-1.,1.,1.], [-1.,1.,1.], [-1.,1.,1.],
            [1.,-1.,1.], [1.,-1.,1.], [1.,-1.,1.],
            [1.,1.,-1.], [1.,1.,-1.], [1.,1.,-1.]
        ])
        
    else:
        raise ValueError('Error: unsupported crystal type = %s in generate_line_config()' % crystal)
    
    nsys = b.shape[0]
    b = b / np.linalg.norm(b, axis=1)[:,None]
    n = n / np.linalg.norm(n, axis=1)[:,None]
    if Rorient is not None:
        Rorient = np.array(Rorient)
        Rorient = Rorient / np.linalg.norm(Rorient, axis=1)[:,None]
        b = np.matmul(b, Rorient.T)
        n = np.matmul(n, Rorient.T)
    
    cell = pyexadis.Cell(Lbox)
    Lmax = np.max(np.linalg.norm(cell.h, axis=0))
    
    if theta is None:
        # Determine the character angles of each dipole
        # such that the densities among slip systems are close.
        # We need to do this because the line length of each dipole
        # depends on the crystal orientation and slip system,
        # with each of which likely to have a different periodicity.
        # Here we first determine the dipole with maximum length.
        ntheta = 19
        theta = 90.0/(ntheta-1)*np.arange(ntheta)
        theta_minlength = np.zeros((nsys, ntheta))
        for isys in range(nsys):
            burg, plane = b[isys], n[isys]
            c = np.array(cell.center())
            # Find character angle that minimizes the line length
            minlength = 1e20
            for t in range(ntheta):
                nodes, segs = [], []
                length = insert_infinite_line(cell, nodes, segs, burg, plane, c,
                                              theta=theta[t], maxseg=maxseg, trial=True)
                theta_minlength[isys,t] = length
        
        # Maximum dipole size among all slip systems
        theta_minlength = np.ma.masked_less(theta_minlength, 0.0)
        minlength = theta_minlength.min(axis=1).filled(-1.0)
        maxlength = np.max(minlength)
        if maxlength > 10*Lmax or np.min(minlength) < 0.0:
            raise ValueError('Error: cannot find appropriate line to insert')
        
        # Select character angle for the slip system that is
        # the closest to the maximum dipole length across
        # all the slip systems
        theta_sys = np.argmin(np.abs(theta_minlength-maxlength), axis=1)
        theta_sys = theta[theta_sys][:,None]
    else:
        theta_sys = np.tile(np.array(theta), (nsys, 1))
    
    # Insert the lines
    if seed > 0: np.random.seed(seed)
    pos = np.random.rand(num_lines, 3)
    pos = np.array(cell.origin) + np.matmul(pos, np.array(cell.h).T)
    ithe = np.random.randint(0, theta_sys.shape[1], num_lines)
    nodes, segs = [], []
    
    for i in range(num_lines):
        isys = i % nsys
        burg, plane = b[isys], n[isys]
        
        idip = np.floor(i/nsys).astype(int) % 2 # alternate sign to create dipoles
        lsign = 1-2*idip
        
        edir = np.cross(plane, burg)
        edir = edir / np.linalg.norm(edir)
        theta = theta_sys[isys,ithe[i-idip*nsys]]
        ldir = np.cos(theta*np.pi/180.0)*burg + np.sin(theta*np.pi/180.0)*edir
        
        if verbose: print(' insert dislocation: b = %.3f %.3f %.3f, n = %.3f %.3f %.3f, theta = %.1f deg' % (*burg, *plane, theta))
        nodes, segs = insert_infinite_line(cell, nodes, segs, burg, plane, pos[i],
                                           linedir=lsign*ldir, maxseg=maxseg)
    
    G = ExaDisNet(cell, nodes, segs)
    return G


def generate_prismatic_config(crystal, Lbox, num_loops, radius, maxseg=-1, Rorient=None, seed=-1, uniform=False):
    """Generate a configuration made of prismatic dislocation loops
    * Dislocation loops are generated by cycling through the list of native
      Burgers vectors for the crystal structure: 6 1/2<110> Burgers vectors
      for fcc and 4 1/2<111> Burgers vectors for bcc.
    Arguments:
    * crystal: crystal structure
    * Lbox: box size or cell object
    * num_loops: number of dislocation loops
    * radius: radius of the loops, or [min_radius, max_radius]
    * maxseg: maximum discretization length of the lines
    * Rorient: crystal orientation matrix
    * seed: seed for random number generation
    * uniform: make the spatial loop distribution close to uniform 
    """    
    #print('generate_prismatic_config()')
    if crystal in ['BCC', 'bcc']:
        b = np.array([[1.,1.,1.],[-1.,1.,1.],[1.,-1.,1.],[1.,1.,-1.]])
    elif crystal in ['FCC', 'fcc']:
        b = np.array([[1.,1.,0.],[-1.,1.,0.],[1.,0.,1.],[-1.,0.,1.],[0.,1.,1.],[0.,-1.,1.]])
    else:
        raise ValueError('Error: unsupported crystal type = %s in generate_prismatic_config()' % crystal)
    
    nburg = b.shape[0]
    b = b / np.linalg.norm(b, axis=1)[:,None]
    
    # Insert the loops
    cell = pyexadis.Cell(Lbox)
    if seed > 0: np.random.seed(seed)
    if uniform:
        # random uniform positions
        ngrid = np.ceil((1.0*num_loops)**(1.0/3.0))
        H = 1.0/ngrid
        x = 0.5*H + H*np.arange(ngrid)
        x, y, z = np.meshgrid(x, x, x)
        p = np.random.permutation(len(x.flatten()))
        x, y, z = x.flatten()[p], y.flatten()[p], z.flatten()[p]
        pos = np.vstack((x, y, z)).T + 0.5*H*(np.random.rand(len(x), 3)-0.5)
    else:
        pos = np.random.rand(num_loops, 3)
    pos = np.array(cell.origin) + np.matmul(pos, np.array(cell.h).T)
    if isinstance(radius, list):
        R = np.random.uniform(radius[0], radius[1], size=(num_loops,))
    else:
        R = radius*np.ones(num_loops)
    
    nodes, segs = [], []
    for i in range(num_loops):
        iburg = i % nburg
        burg = b[iburg]
        nodes, segs = insert_prismatic_loop(crystal, cell, nodes, segs, burg,
                                            R[i], pos[i], maxseg, Rorient)
    
    G = ExaDisNet(cell, nodes, segs)
    return G


def get_segments_end_points(N: DisNetManager) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns the list of dislocation segments end points of the network
    for which the closest image convention is applied to the second end point
    """
    data = N.export_data()
    # cell
    cell = pyexadis.Cell(**data["cell"])
    # nodes
    nodes = data.get("nodes")
    rn = nodes.get("positions")
    # segments
    segs = data.get("segs")
    segsnid = segs.get("nodeids")
    # end points
    r1 = np.array(cell.closest_image(Rref=np.array(cell.center()), R=rn[segsnid[:,0]]))
    r2 = np.array(cell.closest_image(Rref=r1, R=rn[segsnid[:,1]]))
    return r1, r2


def get_segments_length(N: DisNetManager) -> np.ndarray:
    """ Returns the list of dislocation segment lenghts of the network
    """
    r1, r2 = get_segments_end_points(N)
    Lseg = np.linalg.norm(r2-r1, axis=1)
    return Lseg


def dislocation_density(N: DisNetManager, burgmag: float) -> float:
    """ Returns the dislocation density of the network
    """
    len = get_segments_length(N).sum()
    vol = np.abs(np.linalg.det(N.export_data().get("cell")["h"]))
    rho = len/vol/burgmag**2
    return rho


def dislocation_charge(N: DisNetManager) -> np.ndarray:
    """ Returns the dislocation charge (net Nye's tensor) of the network
    """
    r1, r2 = get_segments_end_points(N)
    t = r2-r1
    b = N.export_data()["segs"]["burgers"]
    alpha = np.einsum('ij,ik->jk', b, t)
    return alpha


def read_paradis(datafile: str) -> DisNetManager:
    """ Read dislocation network in ParaDiS format
    """
    G = ExaDisNet().read_paradis(datafile)
    return DisNetManager(G)


def replicate_network(N: DisNetManager, Nrep) -> DisNetManager:
    """ Periodically replicate a dislocation network along the three dimensions
    """
    import copy
    
    if np.isscalar(Nrep): Nrep = Nrep*np.ones(3)
    Nrep = np.array(Nrep).astype(int)
    if np.any(Nrep < 1):
        raise ValueError('replicate_network(): periodic replica (%d,%d,%d) must be at least 1 in each direction' % tuple(Nrep))
    if np.all(Nrep == 1):
        return N
    
    # cell
    data = N.export_data()
    cell0 = pyexadis.Cell(**data["cell"])
    h0 = np.array(cell0.h)
    c1, c2, c3 = h0[:,0], h0[:,1], h0[:,2]
    data["cell"]["h"] *= Nrep
    nodes = data["nodes"]
    nodes0 = copy.deepcopy(nodes)
    num_nodes = nodes["positions"].shape[0]
    
    # periodic nodes replica
    for i3 in range(Nrep[2]):
        for i2 in range(Nrep[1]):
            for i1 in range(Nrep[0]):
                if i1 == 0 and i2 == 0 and i3 == 0: continue
                cnodes = copy.deepcopy(nodes0)
                cnodes["positions"] += i1*c1 + i2*c2 + i3*c3
                for k, v in nodes.items():
                    nodes[k] = np.vstack((nodes[k], cnodes[k]))
    
    # periodic link replica
    segs = data["segs"]
    segs0 = copy.deepcopy(segs)
    nodeids = segs["nodeids"]
    num_segs = nodeids.shape[0]
    p1 = nodes0["positions"][nodeids[:,0]]
    p2 = nodes0["positions"][nodeids[:,1]]
    p2 = np.array(cell0.closest_image(Rref=p1, R=p2))
    repseg = ~np.array(cell0.are_inside(p2), dtype=bool)
    
    nr = 1
    for i3 in range(Nrep[2]):
        for i2 in range(Nrep[1]):
            for i1 in range(Nrep[0]):
                if i1 == 0 and i2 == 0 and i3 == 0: continue
                csegs = copy.deepcopy(segs0)
                csegs["nodeids"] += nr * num_nodes
                for k, v in segs.items():
                    segs[k] = np.vstack((segs[k], csegs[k]))
                nr += 1
    
    # reconnect links across PBC boundaries
    cell = pyexadis.Cell(**data["cell"])
    for s in range(num_segs):
        if not repseg[s]: continue
        for r1 in range(np.prod(Nrep)):
            sr = s + r1 * num_segs
            n1 = segs["nodeids"][sr,0]
            n2 = segs["nodeids"][sr,1] % num_nodes
            # find closest neighbor among PBC images
            n2p = [n2 + r2 * num_nodes for r2 in range(np.prod(Nrep))]
            p1 = nodes["positions"][n1]
            p2 = nodes["positions"][n2p]
            p2 = np.array(cell.closest_image(Rref=p1, R=p2))
            dist = np.linalg.norm(p2-p1, axis=1)
            imin = np.argmin(dist).ravel()[0]
            segs["nodeids"][sr,1] = n2p[imin]
    
    # reset node tags to make sure they are unique
    num_tot_nodes = num_nodes * np.prod(Nrep)
    nodes["tags"] = np.stack((np.zeros(num_tot_nodes), np.arange(num_tot_nodes))).T
    Nnew = DisNetManager(ExaDisNet().import_data(data))
    
    return Nnew


def combine_networks(Nlist) -> DisNetManager:
    """ Combine several DisNetManager into a single network
    """
    if not isinstance(Nlist, list) or len(Nlist) == 0:
        raise ValueError('combine_networks() argument must be a list of DisNetManager')
    
    # combine networks
    for i, Ni in enumerate(Nlist):
        if i == 0:
            data = Ni.export_data()
            nodes, segs = data["nodes"], data["segs"]
            num_nodes = Ni.num_nodes()
        else:
            datai = Ni.export_data()
            if not np.all(datai["cell"]["h"] == data["cell"]["h"]) or \
               not np.all(datai["cell"]["origin"] == data["cell"]["origin"]):
                raise ValueError('combine_networks() networks must use the same cell')
            for k, v in nodes.items():
                nodes[k] = np.vstack((nodes[k], datai["nodes"][k]))
            for k, v in segs.items():
                if k == 'nodeids':
                    segs[k] = np.vstack((segs[k], datai["segs"][k]+num_nodes))
                else:
                    segs[k] = np.vstack((segs[k], datai["segs"][k]))
            num_nodes += Ni.num_nodes()
            
    # reset node tags to make sure they are unique
    nodes["tags"] = np.stack((np.zeros(num_nodes), np.arange(num_nodes))).T
    N = DisNetManager(ExaDisNet().import_data(data))
    return N


def extract_segments(N: DisNetManager, seglist) -> DisNetManager:
    """ Return a new network that contains a subset of segments
    from the input network
    """
    data = N.export_data()
    # keep segments from the list
    segs = data["segs"]
    for k, v in segs.items():
        segs[k] = v[seglist]
    # remove unconnected nodes
    nodelist, nind = np.unique(segs["nodeids"].ravel(), return_inverse=True)
    nodes = data["nodes"]
    for k, v in nodes.items():
        nodes[k] = v[nodelist]
    # update node indices
    segs["nodeids"] = nind[np.arange(segs["nodeids"].size).reshape(-1,2)]
    # create new DisNet
    G = ExaDisNet().import_data(data)
    return DisNetManager(G)


def delete_segments(N: DisNetManager, seglist) -> DisNetManager:
    """ Return a new network in which segments have been deleted
    from the input network
    """
    keeplist = np.setxor1d(seglist, np.arange(N.num_segments()))
    return extract_segments(N, keeplist)


def write_data(N: DisNetManager, datafile: str):
    """ Write dislocation network in ParaDiS format
    """
    N.get_disnet(ExaDisNet).write_data(datafile)


# ============================================================
# Slip system definitions for FCC and BCC crystals
# ============================================================
_FCC_SLIP_SYSTEMS = {
    'planes': np.array([
        [ 1,  1,  1], [ 1,  1,  1], [ 1,  1,  1],
        [-1,  1,  1], [-1,  1,  1], [-1,  1,  1],
        [ 1, -1,  1], [ 1, -1,  1], [ 1, -1,  1],
        [ 1,  1, -1], [ 1,  1, -1], [ 1,  1, -1],
    ], dtype=float),
    'directions': np.array([
        [ 1, -1,  0], [ 1,  0, -1], [ 0,  1, -1],
        [ 1,  1,  0], [ 1,  0,  1], [ 0,  1, -1],
        [ 1,  1,  0], [ 1,  0, -1], [ 0,  1,  1],
        [ 1, -1,  0], [ 1,  0,  1], [ 0,  1,  1],
    ], dtype=float),
}

_BCC_SLIP_SYSTEMS = {
    'planes': np.array([
        [ 0,  1,  1], [ 0,  1,  1], [ 0,  1, -1], [ 0,  1, -1],
        [ 1,  0,  1], [ 1,  0,  1], [ 1,  0, -1], [ 1,  0, -1],
        [ 1,  1,  0], [ 1,  1,  0], [ 1, -1,  0], [ 1, -1,  0],
        [ 2,  1,  1], [-2,  1,  1], [ 1,  2,  1], [-1,  2,  1],
        [ 1,  1,  2], [-1,  1,  2], [ 2,  1, -1], [-2,  1, -1],
        [ 1,  2, -1], [ 1, -2, -1], [ 1,  1, -2], [ 1, -1, -2],
    ], dtype=float),
    'directions': np.array([
        [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1], [ 1,  1,  1],
        [ 1,  1, -1], [-1,  1,  1], [ 1, -1,  1], [ 1,  1,  1],
        [-1,  1,  1], [ 1, -1,  1], [ 1,  1,  1], [ 1,  1, -1],
        [-1,  1,  1], [ 1,  1,  1], [ 1, -1,  1], [ 1,  1, -1],
        [ 1,  1, -1], [ 1, -1,  1], [ 1, -1,  1], [ 1,  1, -1],
        [-1,  1,  1], [ 1,  1, -1], [ 1,  1,  1], [ 1, -1,  1],
    ], dtype=float),
}


def _normalize_vector(v):
    norm = np.linalg.norm(v)
    return v if norm < 1e-10 else v / norm


def _is_parallel(v1, v2, tol=1e-6):
    return np.abs(np.dot(_normalize_vector(v1), _normalize_vector(v2))) > (1.0 - tol)


def _identify_slip_system(burgers, plane, crystal, tol):
    """Return slip system index (1-based). FCC: 1-12/13-other; BCC: 1-24/25-other."""
    b = np.array(burgers, dtype=float)
    pl = np.array(plane, dtype=float)
    systems = _FCC_SLIP_SYSTEMS if crystal == 'FCC' else _BCC_SLIP_SYSTEMS
    n = systems['planes'].shape[0]
    for i in range(n):
        if _is_parallel(pl, systems['planes'][i], tol) and _is_parallel(b, systems['directions'][i], tol):
            return i + 1
    return n + 1  # FCC → 13, BCC → 25


def _burgers_line_angle(burgers, line_vec):
    """Angle between Burgers vector and line direction, clamped to [0, 90] deg."""
    cos_a = np.clip(np.abs(np.dot(_normalize_vector(burgers), _normalize_vector(line_vec))), 0.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def _detect_crystal_type(burgers_vectors):
    """Detect FCC or BCC from Burgers vectors.

    BCC <111>/2: all three |components| similar (ratio min/max > 0.7).
    FCC <110>/2: one component ≈ 0 (ratio min/max < 0.3).
    """
    bcc_votes = 0
    fcc_votes = 0
    for b in burgers_vectors:
        b_abs = np.sort(np.abs(b))  # ascending
        if b_abs[2] < 1e-10:
            continue
        ratio = b_abs[0] / b_abs[2]  # min / max
        if ratio > 0.7:
            bcc_votes += 1
        elif ratio < 0.3:
            fcc_votes += 1
    return 'BCC' if bcc_votes > fcc_votes else 'FCC'


def write_vtk(N: DisNetManager, vtkfile: str, segprops={}, pbc_wrap=True,
              crystal=None, identify_slip_system=True, slip_tol=None, verbose=False):
    """Write dislocation network in vtk format with slip system analysis.

    Parameters
    ----------
    crystal : str or None
        'FCC', 'BCC', or None for auto-detection from Burgers vectors.
    identify_slip_system : bool
        Compute and write SlipSystemID, BurgersLineAngle, DislocationCharacter
        (and SlipSystemFamily for BCC).
    slip_tol : float or None
        Tolerance for slip system matching. Defaults to 1e-6 (FCC) or 1e-3 (BCC).
    verbose : bool
        Print statistics after writing.
    """
    data = N.export_data()

    cell = pyexadis.Cell(**data["cell"])
    cell_origin = np.array(cell.origin)
    cell_center = np.array(cell.center())
    h = np.array(cell.h)
    c = cell_origin + np.array([np.zeros(3), h[0], h[1], h[2], h[0]+h[1],
                                 h[0]+h[2], h[1]+h[2], h[0]+h[1]+h[2]])

    nodes = data.get("nodes")
    rn = nodes.get("positions")

    segs = data.get("segs")
    segsnid = segs.get("nodeids")
    r1 = np.array(cell.closest_image(Rref=cell_center, R=rn[segsnid[:, 0]]))
    r2 = np.array(cell.closest_image(Rref=r1, R=rn[segsnid[:, 1]]))
    b = segs.get("burgers")
    p = segs.get("planes")

    if np.all(np.array(cell.is_periodic()) == 0):
        pbc_wrap = False

    if pbc_wrap:
        eps = 1e-10
        hinv = np.linalg.inv(h)
        is_periodic = np.array(cell.is_periodic())

        def outside_box(pt):
            s = np.matmul(hinv, pt - cell_origin)
            return np.any(((s < -eps) | (s > 1.0 + eps)) & is_periodic)

        def facet_intersection_position(ra, rb, _i):
            s1 = np.matmul(hinv, ra - cell_origin)
            s2 = np.matmul(hinv, rb - cell_origin)
            t = s2 - s1
            t0 = -(s1 - 0.0) / (t + eps)
            t1 = -(s1 - 1.0) / (t + eps)
            sv = np.hstack((t0, t1))
            sv[sv < eps] = 1.0
            facet = np.argmin(sv)
            if sv[facet] < 1.0:
                pos = np.matmul(h, s1 + sv[facet] * t) + cell_origin
                sfacet = sv[facet]
            else:
                facet = -1
                pos = rb
                sfacet = 1.0
            return pos, facet, sfacet

        segsid = []
        rsegs = []
        for i in range(segsnid.shape[0]):
            n1, n2 = segsnid[i]
            ra = np.array(cell.closest_image(Rref=cell_center, R=rn[n1]))
            rb = np.array(cell.closest_image(Rref=ra, R=rn[n2]))
            out = outside_box(rb)
            while out:
                pos, facet, sfacet = facet_intersection_position(ra, rb, i)
                if facet < 0:
                    break
                segsid.append(i)
                rsegs.append([ra, pos])
                ra = pos + (1 - 2 * np.floor(facet / 3)) * (1.0 - 2 * eps) * h[:, facet % 3]
                rb = np.array(cell.closest_image(Rref=ra, R=rn[n2]))
                out = outside_box(rb)
            segsid.append(i)
            rsegs.append([ra, rb])

        segsid = np.array(segsid, dtype=int)
        nsegs = segsid.shape[0]
        rsegs = np.array(rsegs).reshape(-1, 3)
        b = b[segsid]
        p = p[segsid]
        for k, v in segprops.items():
            segprops[k] = v[segsid]
    else:
        nsegs = segsnid.shape[0]
        rsegs = np.hstack((r1, r2)).reshape(-1, 3)
        segsid = np.arange(nsegs)

    line_vectors = rsegs[1::2] - rsegs[0::2]

    # Auto-detect crystal type
    if crystal is None:
        crystal = _detect_crystal_type(b)
    else:
        crystal = crystal.upper()

    if slip_tol is None:
        slip_tol = 1e-3 if crystal == 'BCC' else 1e-6

    if identify_slip_system:
        n_systems = b.shape[0]
        slip_labels = np.zeros(n_systems, dtype=int)
        angles = np.zeros(n_systems, dtype=float)
        for i in range(n_systems):
            slip_labels[i] = _identify_slip_system(b[i], p[i], crystal, slip_tol)
            angles[i] = _burgers_line_angle(b[i], line_vectors[i])

    with open(vtkfile, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Dislocation Network (%s) exported from OpenDiS\n" % crystal)
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        total_points = c.shape[0] + 2 * nsegs
        f.write("POINTS %d double\n" % total_points)
        for pt in c:
            f.write("%.10e %.10e %.10e\n" % (pt[0], pt[1], pt[2]))
        for pt in rsegs:
            f.write("%.10e %.10e %.10e\n" % (pt[0], pt[1], pt[2]))

        total_cells = 1 + nsegs
        f.write("\nCELLS %d %d\n" % (total_cells, 9 + 3 * nsegs))
        f.write("8 0 1 4 2 3 5 7 6\n")
        offset = c.shape[0]
        for i in range(nsegs):
            f.write("2 %d %d\n" % (offset + 2 * i, offset + 2 * i + 1))

        f.write("\nCELL_TYPES %d\n" % total_cells)
        f.write("12\n")
        for _ in range(nsegs):
            f.write("3\n")

        f.write("\nCELL_DATA %d\n" % total_cells)

        f.write("\nVECTORS Burgers double\n")
        f.write("0.0 0.0 0.0\n")
        for vec in b:
            f.write("%.10e %.10e %.10e\n" % (vec[0], vec[1], vec[2]))

        f.write("\nVECTORS Planes double\n")
        f.write("0.0 0.0 0.0\n")
        for vec in p:
            f.write("%.10e %.10e %.10e\n" % (vec[0], vec[1], vec[2]))

        f.write("\nVECTORS LineDirection double\n")
        f.write("0.0 0.0 0.0\n")
        for vec in line_vectors:
            vn = _normalize_vector(vec)
            f.write("%.10e %.10e %.10e\n" % (vn[0], vn[1], vn[2]))

        if identify_slip_system:
            f.write("\nSCALARS SlipSystemID int 1\n")
            f.write("LOOKUP_TABLE default\n")
            f.write("0\n")
            for lbl in slip_labels:
                f.write("%d\n" % lbl)

            if crystal == 'BCC':
                f.write("\nSCALARS SlipSystemFamily int 1\n")
                f.write("LOOKUP_TABLE default\n")
                f.write("0\n")
                for lbl in slip_labels:
                    if 1 <= lbl <= 12:
                        f.write("1\n")   # {110}<111>
                    elif 13 <= lbl <= 24:
                        f.write("2\n")   # {112}<111>
                    else:
                        f.write("3\n")   # other

            f.write("\nSCALARS BurgersLineAngle double 1\n")
            f.write("LOOKUP_TABLE default\n")
            f.write("0.0\n")
            for ang in angles:
                f.write("%.10e\n" % ang)

            f.write("\nSCALARS DislocationCharacter int 1\n")
            f.write("LOOKUP_TABLE default\n")
            f.write("0\n")
            for ang in angles:
                f.write("%d\n" % (0 if ang < 30.0 else (1 if ang < 60.0 else 2)))

        for k, v in segprops.items():
            vals = np.atleast_2d(v.T).T
            if vals.shape[0] != nsegs:
                raise ValueError('segprop "%s" must have the same size as the number of segments' % k)
            dtype_str = "int" if np.issubdtype(vals.dtype, np.integer) else "double"
            f.write("\nSCALARS %s %s %d\n" % (str(k), dtype_str, vals.shape[1]))
            f.write("LOOKUP_TABLE default\n")
            f.write(" ".join(["0"] * vals.shape[1]) + "\n")
            for row in vals:
                f.write(" ".join("%.10e" % val for val in row) + "\n")

    if verbose:
        print("Crystal type: %s" % crystal)
        print("Segments: %d (after PBC wrap)" % nsegs)
        if identify_slip_system:
            if crystal == 'BCC':
                n_110 = int(np.sum((slip_labels >= 1) & (slip_labels <= 12)))
                n_112 = int(np.sum((slip_labels >= 13) & (slip_labels <= 24)))
                n_oth = int(np.sum(slip_labels == 25))
                print("  {110}<111>: %d  {112}<111>: %d  other: %d" % (n_110, n_112, n_oth))
            else:
                n_std = int(np.sum(slip_labels <= 12))
                n_oth = int(np.sum(slip_labels == 13))
                print("  standard FCC: %d  other: %d" % (n_std, n_oth))
            n_screw = int(np.sum(angles < 30.0))
            n_mixed = int(np.sum((angles >= 30.0) & (angles < 60.0)))
            n_edge  = int(np.sum(angles >= 60.0))
            print("  screw: %d  mixed: %d  edge: %d" % (n_screw, n_mixed, n_edge))
        print("VTK written: %s" % vtkfile)
