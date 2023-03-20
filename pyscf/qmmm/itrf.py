#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
QM/MM helper functions that modify the QM methods.
'''

import numpy
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf import df
from pyscf import scf
from pyscf import mcscf
from pyscf import grad
from pyscf.lib import logger
from pyscf.qmmm import mm_mole


def add_mm_charges(scf_method, atoms_or_coords, charges, unit=None):
    '''Embedding the one-electron (non-relativistic) potential generated by MM
    point charges into QM Hamiltonian.

    The total energy includes the regular QM energy, the interaction between
    the nuclei in QM region and the MM charges, and the static Coulomb
    interaction between the electron density and the MM charges. It does not
    include the static Coulomb interactions of the MM point charges, the MM
    energy, the vdw interaction or other bonding/non-bonding effects between
    QM region and MM particles.

    Args:
        scf_method : a HF or DFT object

        atoms_or_coords : 2D array, shape (N,3)
            MM particle coordinates
        charges : 1D array
            MM particle charges
    Kwargs:
        unit : str
            Bohr, AU, Ang (case insensitive). Default is the same to mol.unit

    Returns:
        Same method object as the input scf_method with modified 1e Hamiltonian

    Note:
        1. if MM charge and X2C correction are used together, function mm_charge
        needs to be applied after X2C decoration (.x2c method), eg
        mf = mm_charge(scf.RHF(mol).x2c()), [(0.5,0.6,0.8)], [-0.5]).
        2. Once mm_charge function is applied on the SCF object, it
        affects all the post-HF calculations eg MP2, CCSD, MCSCF etc

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = mm_charge(dft.RKS(mol), [(0.5,0.6,0.8)], [-0.3])
    >>> mf.kernel()
    -101.940495711284
    '''
    mol = scf_method.mol
    if unit is None:
        unit = mol.unit
    mm_mol = mm_mole.create_mm_mol(atoms_or_coords, charges, unit)
    return qmmm_for_scf(scf_method, mm_mol)

# Define method mm_charge for backward compatibility
mm_charge = add_mm_charges

def qmmm_for_scf(scf_method, mm_mol):
    '''Add the potential of MM particles to SCF (HF and DFT) method or CASCI
    method then generate the corresponding QM/MM method for the QM system.

    Args:
        mm_mol : MM Mole object
    '''
    assert (isinstance(scf_method, (scf.hf.SCF, mcscf.casci.CASCI)))

    if isinstance(scf_method, scf.hf.SCF):
        # Avoid to initialize _QMMM twice
        if isinstance(scf_method, _QMMM):
            scf_method.mm_mol = mm_mol
            return scf_method

        method_class = scf_method.__class__

    else:
        if isinstance(scf_method._scf, _QMMM):
            scf_method._scf.mm_mol = mm_mol
            return scf_method

        method_class = scf_method._scf.__class__

    class QMMM(_QMMM, method_class):
        def __init__(self, scf_method, mm_mol):
            self.__dict__.update(scf_method.__dict__)
            self.mm_mol = mm_mol
            self._keys.update(['mm_mol'])

        def dump_flags(self, verbose=None):
            method_class.dump_flags(self, verbose)
            logger.info(self, '** Add background charges for %s **',
                        method_class)
            if self.verbose >= logger.DEBUG:
                logger.debug(self, 'Charge      Location')
                coords = self.mm_mol.atom_coords()
                charges = self.mm_mol.atom_charges()
                for i, z in enumerate(charges):
                    logger.debug(self, '%.9g    %s', z, coords[i])
            return self

        def get_hcore(self, mol=None):
            if mol is None: mol = self.mol
            if getattr(method_class, 'get_hcore', None):
                h1e = method_class.get_hcore(self, mol)
            else:  # DO NOT modify post-HF objects to avoid the MM charges applied twice
                raise RuntimeError('mm_charge function cannot be applied on post-HF methods')

            coords = self.mm_mol.atom_coords()
            charges = self.mm_mol.atom_charges()
            nao = mol.nao
            max_memory = self.max_memory - lib.current_memory()[0]
            blksize = int(min(max_memory*1e6/8/nao**2, 200))
            for i0, i1 in lib.prange(0, charges.size, blksize):
                j3c = mol.intor('int1e_grids', hermi=1, grids=coords[i0:i1])
                h1e += numpy.einsum('kpq,k->pq', j3c, -charges[i0:i1])
            return h1e

        def energy_nuc(self):
            # interactions between QM nuclei and MM particles
            nuc = self.mol.energy_nuc()
            coords = self.mm_mol.atom_coords()
            charges = self.mm_mol.atom_charges()
            for j in range(self.mol.natm):
                q2, r2 = self.mol.atom_charge(j), self.mol.atom_coord(j)
                r = lib.norm(r2-coords, axis=1)
                nuc += q2*(charges/r).sum()
            return nuc

        def nuc_grad_method(self):
            scf_grad = method_class.nuc_grad_method(self)
            return qmmm_grad_for_scf(scf_grad)
        Gradients = nuc_grad_method

    if isinstance(scf_method, scf.hf.SCF):
        return QMMM(scf_method, mm_mol)
    else:  # post-HF methods
        scf_method._scf = QMMM(scf_method._scf, mm_mol).run()
        scf_method.mo_coeff = scf_method._scf.mo_coeff
        scf_method.mo_energy = scf_method._scf.mo_energy
        return scf_method

def add_mm_charges_grad(scf_grad, atoms_or_coords, charges, unit=None):
    '''Apply the MM charges in the QM gradients' method.  It affects both the
    electronic and nuclear parts of the QM fragment.

    Args:
        scf_grad : a HF or DFT gradient object (grad.HF or grad.RKS etc)
            Once the add_mm_charges_grad was applied, it affects all post-HF
            calculations eg MP2, CCSD, MCSCF etc
        coords : 2D array, shape (N,3)
            MM particle coordinates
        charges : 1D array
            MM particle charges
    Kwargs:
        unit : str
            Bohr, AU, Ang (case insensitive). Default is the same to mol.unit

    Returns:
        Same gradeints method object as the input scf_grad method

    Examples:

    >>> from pyscf import gto, scf, grad
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = mm_charge(scf.RHF(mol), [(0.5,0.6,0.8)], [-0.3])
    >>> mf.kernel()
    -101.940495711284
    >>> hfg = mm_charge_grad(grad.hf.RHF(mf), coords, charges)
    >>> hfg.kernel()
    [[-0.25912357 -0.29235976 -0.38245077]
     [-1.70497052 -1.89423883  1.2794798 ]]
    '''
    assert (isinstance(scf_grad, grad.rhf.Gradients))
    mol = scf_grad.mol
    if unit is None:
        unit = mol.unit
    mm_mol = mm_mole.create_mm_mol(atoms_or_coords, charges, unit)
    mm_grad = qmmm_grad_for_scf(scf_grad)
    mm_grad.base.mm_mol = mm_mol
    return mm_grad

# Define method mm_charge_grad for backward compatibility
mm_charge_grad = add_mm_charges_grad

def qmmm_grad_for_scf(scf_grad):
    '''Add the potential of MM particles to SCF (HF and DFT) object and then
    generate the corresponding QM/MM gradients method for the QM system.
    '''
    if getattr(scf_grad.base, 'with_x2c', None):
        raise NotImplementedError('X2C with QM/MM charges')

    # Avoid to initialize _QMMMGrad twice
    if isinstance(scf_grad, _QMMMGrad):
        return scf_grad

    assert (isinstance(scf_grad.base, scf.hf.SCF) and
           isinstance(scf_grad.base, _QMMM))

    grad_class = scf_grad.__class__
    class QMMM(_QMMMGrad, grad_class):
        def __init__(self, scf_grad):
            self.__dict__.update(scf_grad.__dict__)

        def dump_flags(self, verbose=None):
            grad_class.dump_flags(self, verbose)
            logger.info(self, '** Add background charges for %s **', grad_class)
            if self.verbose >= logger.DEBUG1:
                logger.debug1(self, 'Charge      Location')
                coords = self.base.mm_mol.atom_coords()
                charges = self.base.mm_mol.atom_charges()
                for i, z in enumerate(charges):
                    logger.debug1(self, '%.9g    %s', z, coords[i])
            return self

        def get_hcore(self, mol=None):
            ''' (QM 1e grad) + <-d/dX i|q_mm/r_mm|j>'''
            if mol is None: mol = self.mol
            coords = self.base.mm_mol.atom_coords()
            charges = self.base.mm_mol.atom_charges()

            nao = mol.nao
            max_memory = self.max_memory - lib.current_memory()[0]
            blksize = int(min(max_memory*1e6/8/nao**2/3, 200))
            g_qm = grad_class.get_hcore(self, mol)
            for i0, i1 in lib.prange(0, charges.size, blksize):
                j3c = mol.intor('int1e_grids_ip', grids=coords[i0:i1])
                g_qm += numpy.einsum('ikpq,k->ipq', j3c, charges[i0:i1])
            return g_qm

        def grad_nuc(self, mol=None, atmlst=None):
            if mol is None: mol = self.mol
            coords = self.base.mm_mol.atom_coords()
            charges = self.base.mm_mol.atom_charges()

            g_qm = grad_class.grad_nuc(self, mol, atmlst)
# nuclei lattice interaction
            g_mm = numpy.empty((mol.natm,3))
            for i in range(mol.natm):
                q1 = mol.atom_charge(i)
                r1 = mol.atom_coord(i)
                r = lib.norm(r1-coords, axis=1)
                g_mm[i] = -q1 * numpy.einsum('i,ix,i->x', charges, r1-coords, 1/r**3)
            if atmlst is not None:
                g_mm = g_mm[atmlst]
            return g_qm + g_mm
    return QMMM(scf_grad)

# A tag to label the derived class
class _QMMM:
    pass
class _QMMMGrad:
    pass

# Inject QMMM interface wrapper to other modules
scf.hf.SCF.QMMM = mm_charge
mcscf.casci.CASCI.QMMM = mm_charge
grad.rhf.Gradients.QMMM = mm_charge_grad

if __name__ == '__main__':
    from pyscf import scf, cc, grad
    mol = gto.Mole()
    mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                   H                 -0.00000000   -0.84695236    0.59109389
                   H                 -0.00000000    0.89830571    0.52404783 '''
    mol.basis = 'cc-pvdz'
    mol.build()

    coords = [(0.5,0.6,0.8)]
    #coords = [(0.0,0.0,0.0)]
    charges = [-0.5]
    mf = mm_charge(scf.RHF(mol), coords, charges)
    print(mf.kernel()) # -76.3206550372

    g = mf.nuc_grad_method().kernel()
    mfs = mf.as_scanner()
    e1 = mfs(''' O                  0.00100000    0.00000000   -0.11081188
             H                 -0.00000000   -0.84695236    0.59109389
             H                 -0.00000000    0.89830571    0.52404783 ''')
    e2 = mfs(''' O                 -0.00100000    0.00000000   -0.11081188
             H                 -0.00000000   -0.84695236    0.59109389
             H                 -0.00000000    0.89830571    0.52404783 ''')
    print((e1 - e2)/0.002 * lib.param.BOHR, g[0,0])

    mycc = cc.ccsd.CCSD(mf)
    ecc, t1, t2 = mycc.kernel() # ecc = -0.228939687075

    g = mycc.nuc_grad_method().kernel()
    ccs = mycc.as_scanner()
    e1 = ccs(''' O                  0.00100000    0.00000000   -0.11081188
             H                 -0.00000000   -0.84695236    0.59109389
             H                 -0.00000000    0.89830571    0.52404783 ''')
    e2 = ccs(''' O                 -0.00100000    0.00000000   -0.11081188
             H                 -0.00000000   -0.84695236    0.59109389
             H                 -0.00000000    0.89830571    0.52404783 ''')
    print((e1 - e2)/0.002 * lib.param.BOHR, g[0,0])

