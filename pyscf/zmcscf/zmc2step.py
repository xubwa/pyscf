#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import copy, scipy, time, numpy
from functools import reduce
import pyscf.lib.logger as logger
from pyscf.zmcscf import gzcasci, zmc_ao2mo
from pyscf.shciscf import shci
#from pyscf.mcscf import mc1step
#from pyscf.mcscf import casci
from pyscf import __config__, lib, ao2mo
#from pyscf.mcscf.casci import get_fock, cas_natorb, canonicalize
from pyscf.mcscf import mc_ao2mo
from pyscf.mcscf import chkfile
from scipy.linalg import expm as expmat 
from scipy.sparse.linalg import LinearOperator, minres, gmres
from pyscf.soscf import ciah
import pyscf.df
from pyscf import mcscf
from memory_profiler import profile
import cProfile

def bracket(A,B):
    return numpy.dot(A,B)-numpy.dot(B,A)
def derivOfExp(a,dA, maxT=50):
    fact = 1.0
    deriv = 1.*dA
    bra = 1.*dA
    for i in range(maxT):
        bra = bracket(a, bra)
        fact *= -1./(i+2.)
        deriv += fact*bra

    return deriv

def arnoldi(A, Q, H, k):
    Q[:,k+1] = A(Q[:,k])
    for i in range(k+1):
        H[i, k] = numpy.dot(Q[:,i], Q[:,k+1])
        Q[:,k+1] = Q[:,k+1] - H[i,k] * Q[:,i]

    H[k+1,k] = numpy.linalg.norm(Q[:,k+1])
    Q[:,k+1] = Q[:,k+1]/H[k+1,k]

def GMRES(A, b, x, max_iter=20, conv=1.e-5):
    from numpy.linalg import norm
    m, n = max_iter, b.shape[0]

    r = b - A(x)
    b_norm, r_norm = norm(b), norm(r)
    error = r_norm/b_norm

    e1 = numpy.zeros((m+1,)) 
    e1[0] = r_norm

    Q, H = numpy.zeros((n,m)), numpy.zeros((m+1,m))
    Q[:,0] = r/r_norm
    beta = r_norm * e1

    for k in range(max_iter-1):
        arnoldi(A, Q, H, k)

        result = numpy.linalg.lstsq(H, e1, rcond=None)[0]
        err = norm(numpy.dot(H,result)-e1)
        #print (k, err)
        x = numpy.dot(Q, result)

    #print (norm(A(x)-b))
    return x, err

   
def conjugateGradient(A, b, x, max_iter=10, conv=1.e-4):
    r = b - A(x)
    p = 1*r
    rsold = numpy.dot(r,r)

    for i in range(max_iter):
        Ap = A(p)
        alpha = rsold / numpy.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = numpy.dot(r,r)
        if (rsnew)**0.5 < conv:
              break
        p = r + (rsnew / rsold) * p
        print ("cg, ", i, rsnew**0.5)
        if (i == 5):
            #reset it
            r = b - A(x)
            p = 1*r
            rsnew = numpy.dot(r,r)
        rsold = rsnew
    return x, rsnew**0.5
def _sqrt(a, tol=1e-14):
    e, v = numpy.linalg.eigh(a)
    idx = e > tol
    return numpy.dot(v[:,idx]*numpy.sqrt(e[idx]), v[:,idx].T.conj())

def get_jk_df(cderi, mo_coeff, dm=None, with_j=True, with_k=True):
    norb = mo_coeff.shape[1]
    mo = mo_coeff

    if dm is None:
        dm_ao = numpy.dot(mo, mo.T.conj())
    else:
        dm_sqrt = _sqrt(dm)
        dm_ao = reduce(numpy.dot, (mo.conj(), dm, mo.T)).conj()
        mo = lib.einsum('ij,jk->ik', mo_coeff, dm_sqrt)

    if (with_j):
        j1 = lib.einsum('Lij,ji->L', cderi, dm_ao)
        j = lib.einsum('Lij,L->ij', cderi, j1)
    if (with_k):
        cderi_half_trans = lib.einsum('pij,jk->pik', cderi, mo)
        k = lib.einsum('pik,pkj->ij', cderi_half_trans, cderi_half_trans.transpose(0,2,1).conj())
    return j,k

def sph2spinor(c, a):
    return reduce(numpy.dot, (c.T.conj(), a, c))

def spinor2sph(c, a):
    return reduce(numpy.dot, (c, a, c.T.conj()))

def get_jk_df_scalar(mol, cderi, mo_coeff, dm=None, with_j=True, with_k=True):
    mo = mo_coeff
    ca, cb = mol.sph2spinor_coeff()

    if dm is None:
        dm_ao = numpy.dot(mo, mo.T.conj())
    else:
        dm_sqrt = _sqrt(dm)
        dm_ao = reduce(numpy.dot, (mo.conj(), dm, mo.T)).conj()
        mo = lib.einsum('ij,jk->ik', mo_coeff, dm_sqrt)

    if (with_j):
        dm_ao_scalar = spinor2sph(ca, dm_ao) + spinor2sph(cb, dm_ao)
        j1 = lib.einsum('Lij,ji->L', cderi, dm_ao_scalar)
        j = lib.einsum('Lij,L->ij', cderi, j1)
        vj = sph2spinor(ca, j) + sph2spinor(cb, j)

    if (with_k):
        mo_coeff_half_trans_alpha = numpy.dot(ca, mo)
        mo_coeff_half_trans_beta = numpy.dot(cb, mo)
        cderi_scalar_half_trans_alpha = lib.einsum('pij,jk->pik', cderi,       mo_coeff_half_trans_alpha)
        cderi_scalar_half_trans_beta = lib.einsum('pij,jk->pik', cderi,        mo_coeff_half_trans_beta)
        vkaa = lib.einsum('pik,pkj->ij', cderi_scalar_half_trans_alpha,       cderi_scalar_half_trans_alpha.transpose(0,2,1).conj())
        vkab = lib.einsum('pik,pkj->ij', cderi_scalar_half_trans_alpha,       cderi_scalar_half_trans_beta.transpose(0,2,1).conj())
        vkba = lib.einsum('pik,pkj->ij', cderi_scalar_half_trans_beta,        cderi_scalar_half_trans_alpha.transpose(0,2,1).conj())
        vkbb = lib.einsum('pik,pkj->ij', cderi_scalar_half_trans_beta,        cderi_scalar_half_trans_beta.transpose(0,2,1).conj())
        vk = (sph2spinor(ca, vkaa) + sph2spinor(cb, vkbb) + reduce(numpy.dot, (ca.T.conj(), vkab, cb)) + reduce(numpy.dot, (cb.T.conj(), vkba, ca)) )
    return vj, vk

def kernel(casscf, mo_coeff, tol=1e-7, conv_tol_grad=None,
           ci0=None, callback=None, verbose=None, dump_chk=True):
    if verbose is None:
        verbose = casscf.verbose
    if callback is None:
        callback = casscf.callback

    log = logger.Logger(casscf.stdout, verbose)
    cput0 = (time.clock(), time.time())
    log.debug('Start 2-step ZCASSCF')

    mo = mo_coeff
    nmo = mo.shape[1]
    ncore = casscf.ncore
    ncas = casscf.ncas
    nocc = ncore + ncas

    eris = None
    e_tot, e_cas, fcivec = casscf.casci(mo, ci0, eris, log, locals())
    log.timer('CASCI finished')
    if ncas == nmo and not casscf.internal_rotation:
        if casscf.canonicalization:
            log.debug('CASSCF canonicalization')
            mo, fcivec, mo_energy = casscf.canonicalize(mo, fcivec, eris,
                                                        casscf.sorting_mo_energy,
                                                        casscf.natorb, verbose=log)
        else:
            mo_energy = None
        return True, e_tot, e_cas, fcivec, mo, mo_energy

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(tol)
        logger.info(casscf, 'Set conv_tol_grad to %g', conv_tol_grad)
    conv_tol_ddm = conv_tol_grad * 3
    conv = False
    de, elast = e_tot, e_tot
    totmicro = totinner = 0
    casdm1 = 0
    r0 = None

    t2m = t1m = log.timer('Initializing 2-step CASSCF', *cput0)
    imacro = 0
    while not conv and imacro < casscf.max_cycle_macro:
        imacro += 1
        njk = 0
        casdm1_old = casdm1
        casdm1, casdm2 = casscf.fcisolver.make_rdm12Frombin(fcivec, ncas, casscf.nelecas)
        norm_ddm = numpy.linalg.norm(casdm1 - casdm1_old)
        t3m = log.timer('update CAS DM', *t2m)

        max_cycle_micro = casscf.micro_cycle_scheduler(locals())
        #max_stepsize = casscf.max_stepsize_scheduler(locals())
 
        print("macro iter %d, E=%16.12g " %(imacro, e_tot), end='')
        mo, gorb, njk, norm_gorb0 = casscf.optimizeOrbs(mo, lambda:casdm1, lambda:casdm2, imacro <= 2, 
                                        eris, r0, conv_tol_grad*0.3, log)
        norm_gorb = numpy.linalg.norm(gorb)
        totinner += njk

        t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)
        t3m = t2m
        e_tot, e_cas, fcivec = casscf.casci(mo, fcivec, eris, log, locals())
        log.timer('CASCI solver', *t3m)

        de, elast = e_tot - elast, e_tot
        if (abs(de) < tol and
            norm_gorb < conv_tol_grad and norm_ddm < conv_tol_ddm):
            conv = True
        else:
            elast = e_tot

        ###FIX THIS###
        #if dump_chk:
        #casscf.dump_chk(locals())

        if callable(callback):
            callback(locals())

    if conv:
        log.info('2-step CASSCF converged in %d macro (%d JK %d micro) steps',
                 imacro, totinner, totmicro)
    else:
        log.info('2-step CASSCF not converged, %d macro (%d JK %d micro) steps',
                 imacro, totinner, totmicro)

    if casscf.canonicalization:
        log.info('CASSCF canonicalization')
        mo, fcivec, mo_energy = \
                casscf.canonicalize(mo, fcivec, eris, casscf.sorting_mo_energy,
                                    casscf.natorb, casdm1, log)
        if casscf.natorb and dump_chk: # dump_chk may save casdm1
            occ, ucas = casscf._eig(-casdm1, ncore, nocc)
            casdm1 = numpy.diag(-occ)

    if dump_chk:
        casscf.dump_chk(locals())

    log.timer('2-step CASSCF', *cput0)
    return conv, e_tot, e_cas, fcivec, mo, mo_energy

class ZCASSCF(gzcasci.GZCASCI):
    __doc__ = gzcasci.GZCASCI.__doc__ + '''CASSCF
    Extra attributes for CASSCF:
        conv_tol : float
            Converge threshold.  Default is 1e-7
        conv_tol_grad : float
            Converge threshold for CI gradients and orbital rotation gradients.
            Default is 1e-4
        max_cycle_macro : int
            Max number of macro iterations.  Default is 50.
        internal_rotation: bool.
            if the CI solver is not FCI then active-active rotations are not redundant.
            Default(True)
        chkfile : str
            Checkpoint file to save the intermediate orbitals during the CASSCF optimization.
            Default is the checkpoint file of mean field object.
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            envrionment.
    Saved results
        e_tot : float
            Total MCSCF energy (electronic energy plus nuclear repulsion)
        e_cas : float
            CAS space FCI energy
        ci : ndarray
            CAS space FCI coefficients
        mo_coeff : ndarray (MxM, but the number of active variables is BB are just first N(=ncore+nact) columns)
            Optimized CASSCF orbitals coefficients. When canonicalization is
            specified, the returned orbitals make the general Fock matrix
            (Fock operator on top of MCSCF 1-particle density matrix)
            diagonalized within each subspace (core, active, external).
            If natorb (natural orbitals in active space) is specified,
            the active segment of the mo_coeff is natural orbitls.
        mo_energy : ndarray
            Diagonal elements of general Fock matrix (in mo_coeff
            representation).
    Examples:
    ********CHANGE THIS EXAMPLE***********
    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = mcscf.CASSCF(mf, 6, 6)
    >>> mc.kernel()[0]
    -109.044401882238134
    '''

# the max orbital rotation and CI increment, prefer small step size
    max_cycle_macro = getattr(__config__, 'mcscf_mc1step_CASSCF_max_cycle_macro', 50)
    max_cycle_micro = getattr(__config__, 'mcscf_mc1step_CASSCF_max_cycle_micro', 4)
    conv_tol = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_conv_tol', 1e-7)
    conv_tol_grad = getattr(__config__, 'mcscf_zmc2step_ZCASSCF_conv_tol_grad', None)

    ah_level_shift = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_level_shift', 1e-8)
    ah_conv_tol = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_conv_tol', 1e-12)
    ah_max_cycle = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_max_cycle', 30)
    ah_lindep = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_lindep', 1e-14)
# * ah_start_tol and ah_start_cycle control the start point to use AH step.
#   In function rotate_orb_cc, the orbital rotation is carried out with the
#   approximate aug_hessian step after a few davidson updates of the AH eigen
#   problem.  Reducing ah_start_tol or increasing ah_start_cycle will delay
#   the start point of orbital rotation.
# * We can do early ah_start since it only affect the first few iterations.
#   The start tol will be reduced when approach the convergence point.
# * Be careful with the SYMMETRY BROKEN caused by ah_start_tol/ah_start_cycle.
#   ah_start_tol/ah_start_cycle actually approximates the hessian to reduce
#   the J/K evaluation required by AH.  When the system symmetry is higher
#   than the one given by mol.symmetry/mol.groupname,  symmetry broken might
#   occur due to this approximation,  e.g.  with the default ah_start_tol,
#   C2 (16o, 8e) under D2h symmetry might break the degeneracy between
#   pi_x, pi_y orbitals since pi_x, pi_y belong to different irreps.  It can
#   be fixed by increasing the accuracy of AH solver, e.g.
#               ah_start_tol = 1e-8;  ah_conv_tol = 1e-10
# * Classic AH can be simulated by setting eg
#               ah_start_tol = 1e-7
#               max_stepsize = 1.5
#               ah_grad_trust_region = 1e6
# ah_grad_trust_region allow gradients being increased in AH optimization
    ah_start_tol = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_start_tol', 2.5)
    ah_start_cycle = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_start_cycle', 3)
    ah_grad_trust_region = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_grad_trust_region', 3.0)
    internal_rotation = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_internal_rotation', True)
    kf_interval = getattr(__config__, 'mcscf_mc1step_CASSCF_kf_interval', 4)
    kf_trust_region = getattr(__config__, 'mcscf_mc1step_CASSCF_kf_trust_region', 3.0)

    ao2mo_level = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_ao2mo_level', 2)
    natorb = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_natorb', False)
    canonicalization = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_canonicalization', True)
    sorting_mo_energy = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_sorting_mo_energy', False)

    def __init__(self, mf_or_mol, ncas, nelecas, auxbasis = 'weigend+etb', ncore=None, frozen=None):
        gzcasci.GZCASCI.__init__(self, mf_or_mol, ncas, nelecas, ncore)
        self.frozen = frozen
        self.hcore = self._scf.get_hcore()
        self.callback = None
        self.chkfile = self._scf.chkfile

        self.fcisolver.max_cycle = getattr(__config__,
                                           'zmcscf_zmc2step_ZCASSCF_fcisolver_max_cycle', 50)
        self.fcisolver.conv_tol = getattr(__config__,
                                          'zmcscf_zmc2step_ZCASSCF_fcisolver_conv_tol', 1e-8)

##################################################
# don't modify the following attributes, they are not input options
        self.e_tot = None
        self.e_cas = None
        self.ci = None
        self.mo_coeff = self._scf.mo_coeff
        self.mo_energy = self._scf.mo_energy
        self.converged = False
        self._max_stepsize = None

        #calculate the integrals
        #self.cderi = pyscf.df.r_incore.cholesky_eri(self.mol, auxbasis=auxbasis, int3c='int3c2e_spinor')
        self.cderi_scalar = pyscf.df.incore.cholesky_eri(self.mol, auxbasis='weigend+etb', int3c='int3c2e', aosym='s1').reshape(-1, self.mol.nao, self.mol.nao)
        #self.cderi.shape = (self.cderi.shape[0], self.mo_coeff.shape[0], self.mo_coeff.shape[1])
        #print ("shape", self.cderi.shape)
        keys = set(('max_cycle_macro',
                    'conv_tol', 'conv_tol_grad',
                    'bb_conv_tol', 'bb_max_cycle', 
                    'internal_rotation',
                     'fcisolver_max_cycle',
                    'fcisolver_conv_tol', 'natorb', 'canonicalization',
                    'sorting_mo_energy', 'scale_restoration'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        ncore = self.ncore
        ncas = self.ncas
        nvir = self.mo_coeff.shape[1] - ncore - ncas
        log.info('CAS (%de+%de, %do), ncore = %d, nvir = %d', \
                 self.nelecas[0], self.nelecas[1], ncas, ncore, nvir)
        assert(nvir >= 0 and ncore >= 0 and ncas >= 0)
        if self.frozen is not None:
            log.info('frozen orbitals %s', str(self.frozen))
        log.info('max_cycle_macro = %d', self.max_cycle_macro)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_grad = %s', self.conv_tol_grad)
        log.info('natorb = %s', self.natorb)
        log.info('canonicalization = %s', self.canonicalization)
        log.info('sorting_mo_energy = %s', self.sorting_mo_energy)
        log.info('ao2mo_level = %d', self.ao2mo_level)
        log.info('chkfile = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        log.info('internal_rotation = %s', self.internal_rotation)
        if getattr(self.fcisolver, 'dump_flags', None):
            self.fcisolver.dump_flags(self.verbose)
        if self.mo_coeff is None:
            log.error('Orbitals for CASCI are not specified. The relevant SCF '
                      'object may not be initialized.')

        if (getattr(self._scf, 'with_solvent', None) and
            not getattr(self, 'with_solvent', None)):
            log.warn('''Solvent model %s was found at SCF level but not applied to the CASSCF object.
The SCF solvent model will not be applied to the current CASSCF calculation.
To enable the solvent model for CASSCF, the following code needs to be called
        from pyscf import solvent
        mc = mcscf.CASSCF(...)
        mc = solvent.ddCOSMO(mc)
''',
                     self._scf.with_solvent.__class__)
        return self


    def kernel(self, mo_coeff=None, ci0=None, callback=None, _kern=kernel):
        '''
        Returns:
            Five elements, they are
            total energy,
            active space CI energy,
            the active space FCI wavefunction coefficients or DMRG wavefunction ID,
            the MCSCF canonical orbital coefficients,
            the MCSCF canonical orbital coefficients.
        They are attributes of mcscf object, which can be accessed by
        .e_tot, .e_cas, .ci, .mo_coeff, .mo_energy
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else: # overwrite self.mo_coeff because it is needed in many methods of this class
            self.mo_coeff = mo_coeff
        if callback is None: callback = self.callback

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.converged, self.e_tot, self.e_cas, self.ci, \
                self.mo_coeff, self.mo_energy = \
                _kern(self, mo_coeff,
                      tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
                      ci0=ci0, callback=callback, verbose=self.verbose)
        logger.note(self, 'CASSCF energy = %.15g', self.e_tot)
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def mc1step(self, mo_coeff=None, ci0=None, callback=None):
        return self.kernel(mo_coeff, ci0, callback)

    def mc2step(self, mo_coeff=None, ci0=None, callback=None):
        from pyscf.mcscf import mc2step
        return self.kernel(mo_coeff, ci0, callback, mc2step.kernel)

    def micro_cycle_scheduler(self, envs):
        return self.max_cycle_micro

        #log_norm_ddm = numpy.log(envs['norm_ddm'])
        #return max(self.max_cycle_micro, int(self.max_cycle_micro-1-log_norm_ddm))

    def ao2mo(self, mo_coeff=None, level=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if level is None: level=self.ao2mo_level      
        return zmc_ao2mo._ERIS(self, mo_coeff, level=level)

    def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        log = logger.new_logger(self, verbose)

        fcasci = copy.copy(self)
        fcasci.ao2mo = self.get_h2cas

        e_tot, e_cas, fcivec = gzcasci.kernel(fcasci, mo_coeff, ci0, log)
        if not isinstance(e_cas, (float, numpy.number)):
            raise RuntimeError('Multiple roots are detected in fcisolver.  '
                               'CASSCF does not know which state to optimize.\n'
                               'See also  mcscf.state_average  or  mcscf.state_specific  for excited states.')
        elif numpy.ndim(e_cas) != 0:
            # This is a workaround for external CI solver compatibility.
            e_cas = e_cas[0]

        if envs is not None and log.verbose >= logger.INFO:
            log.debug('CAS space CI energy = %.15g', e_cas)


            if 'imicro' in envs:  # Within CASSCF iteration
                log.info('macro iter %d (%d JK  %d micro), '
                         'CASSCF E = %.15g  dE = %.8g',
                          envs['imacro'], envs['njk'], envs['imicro'],
                          e_tot, e_tot-envs['elast'])
                if 'norm_gci' in envs:
                    log.info('               |grad[o]|=%5.3g  '
                             '|grad[c]|= %s  |ddm|=%5.3g',
                             envs['norm_gorb0'],
                             envs['norm_gci'], envs['norm_ddm'])
                else:
                    log.info('               |grad[o]|=%5.3g  |ddm|=%5.3g',
                             envs['norm_gorb0'], envs['norm_ddm'])
            else:  # Initialization step
                log.info('CASCI E = %.15g', e_tot)
        return e_tot, e_cas, fcivec

    def dump_chk(self, envs):
        if not self.chkfile:
            return self

        ncore = self.ncore
        nocc = ncore + self.ncas
        if 'mo' in envs:
            mo_coeff = envs['mo']
        else:
            mo_coeff = envs['mo_coeff']
        mo_occ = numpy.zeros(mo_coeff.shape[1])
        mo_occ[:ncore] = 2
        if self.natorb:
            occ = self._eig(-envs['casdm1'], ncore, nocc)[0]
            mo_occ[ncore:nocc] = -occ
        else:
            mo_occ[ncore:nocc] = envs['casdm1'].diagonal().real
# Note: mo_energy in active space =/= F_{ii}  (F is general Fock)
        if 'mo_energy' in envs:
            mo_energy = envs['mo_energy']
        else:
            mo_energy = 'None'
        chkfile.dump_mcscf(self, self.chkfile, 'mcscf', envs['e_tot'],
                           mo_coeff, ncore, self.ncas, mo_occ,
                           mo_energy, envs['e_cas'], None, envs['casdm1'],
                           overwrite_mol=False)
        return self

    def update_from_chk(self, chkfile=None):
        if chkfile is None: chkfile = self.chkfile
        self.__dict__.update(lib.chkfile.load(chkfile, 'mcscf'))
        return self
    update = update_from_chk
                         
    
    #Fully uses AO and is currently not efficient because AO 
    #integrals are assumed to be available cheaply
    def calcGradAO(self, mo, casdm1, casdm2):
        hcore = self.hcore #self._scf.get_hcore() 
        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact

        gradC2 = 0*mo
        moc = mo[:,:ncore]
        moa = mo[:,ncore:nocc]

        #ecore 
        dmcas = reduce(numpy.dot, (moa.conj(), casdm1(), moa.T)).conj()
        dmcore = numpy.dot(moc, moc.conj().T)

        jc,kc = self._scf.get_jk(self.cderi, dm=dmcore)        
        ja,ka = self._scf.get_jk(self.cderi, dm=dmcas)        
        
        gradC2[:,:ncore] = numpy.dot( (hcore + (jc-kc)+ja-ka) , moc)  

        gradC2[:,ncore:nocc] = reduce(numpy.dot, ( (hcore + jc - kc)  , moa, casdm1().T))

        ###THIS IS THE BIT WE NEED
        '''
        eri_ao_sp = mol.intor('int2e_spinor', aosym='s1')
        j1 = lib.einsum('wxyz, wp->pxyz', eri_ao_sp, moa.conj())
        jaapp = lib.einsum('pxyz, xq->pqyz', j1, moa)
        jaaap = lib.einsum('pqyz, zs->pqys', jaapp, moa)
        gradC2[:,ncore:nocc] += lib.einsum('pqys,prqs->yr', jaaap, casdm2())
        '''
        ####### FOR GRADIENT

        ###THIS BIT WILL BE EXPENSIVE       
        eripaaa = numpy.zeros((nmo, nact, nact,nact), dtype = complex)
        for i in range(nact):
            for j in range(i+1):
                dm = lib.einsum('x,y->xy',moa[:,i], moa[:,j].conj())
                j1 = self._scf.get_j(self.mol, dm = dm, hermi=0)
                j1 = numpy.triu(j1)
                j1 = j1 + j1.T - numpy.diag(numpy.diag(j1))
                eripaaa[:,:,i,j] = numpy.dot(j1,moa)
                if (i != j):
                    eripaaa[:,:,j,i] = numpy.dot(j1.conj().T,moa)
        gradC2[:,ncore:nocc] += lib.einsum('ypqr,sqpr->ys', eripaaa, casdm2())

        gradC2 = numpy.dot(mo.conj().T, gradC2)
        return 2*gradC2
       
    def calcGradDFScalar(self, mo, casdm1, casdm2):
        start = time.clock(), time.time()
        cderi = self.cderi_scalar

        hcore = self.hcore #self._scf.get_hcore()

        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact
        nuc_energy = self.energy_nuc()

        Grad = numpy.zeros((nmo, nmo), dtype=complex)
        moc, moa = mo[:,:ncore], mo[:,ncore:nocc]
        j,k=get_jk_df_scalar(self.mol, cderi, moc)
        ja,ka=get_jk_df_scalar(self.mol, cderi, moa, dm = casdm1())

        hcore = reduce(numpy.dot, (mo.conj().T, hcore , mo))
        Fc =  (hcore + reduce(numpy.dot, (mo.conj().T, (j-k), mo)))
        Grad[:,:ncore] = hcore[:,:ncore] + reduce(numpy.dot, (mo.conj().T, j + ja - k - ka, moc))
        
        Grad[:,ncore:nocc] = numpy.einsum('sp,qp->sq', Fc[:,ncore:nocc], casdm1())

        ca, cb = self.mol.sph2spinor_coeff()
        moa_alpha = numpy.dot(ca, moa)
        moa_beta  = numpy.dot(cb, moa)
        Lrq = lib.einsum('Lxj,xi,ja->Lia', cderi, ca.conj(), moa_alpha) \
            + lib.einsum('Lxj,xi,ja->Lia', cderi, cb.conj(), moa_beta)
        Lpq = lib.einsum('Lxa,xb->Lba', Lrq, moa.conj())
        Lrq = lib.einsum('Lxa,xy->Lya', Lrq, mo.conj())
        paaa = lib.einsum('Lxa,Lcd->xacd', Lrq, Lpq)
        Grad[:,ncore:nocc]+= lib.einsum('ruvw,tvuw->rt', paaa, casdm2())

        E = nuc_energy + 0.5*numpy.sum((hcore+Fc).diagonal()[:ncore]) 
        E += numpy.einsum('tu, tu', Fc[ncore:nocc, ncore:nocc], casdm1())
        E += 0.5*lib.einsum('ruvw,rvuw', paaa[ncore:nocc], casdm2())
        #print("calc grad df from scalar cderi", time.clock() - start[0], time.time()-start[1]) 
        return 2*Grad, E

    def calcGradDF(self, mo, casdm1, casdm2):
        start = time.clock(), time.time()
        #return self.calcGrad(mo, casdm1, casdm2)
        hcore = self.hcore #_scf.get_hcore()

        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact
        nuc_energy = self.energy_nuc()

        Grad = numpy.zeros((nmo, nmo), dtype=complex)
        moc, moa = mo[:,:ncore], mo[:,ncore:nocc]
        j,k=get_jk_df(self.cderi, moc)
        ja,ka=get_jk_df(self.cderi, moa, dm = casdm1())

        hcore = reduce(numpy.dot, (mo.conj().T, hcore , mo))
        Fc =  (hcore + reduce(numpy.dot, (mo.conj().T, (j-k), mo)))
        Grad[:,:ncore] = hcore[:,:ncore] + reduce(numpy.dot, (mo.conj().T, j + ja - k - ka, moc))
        
        Grad[:,ncore:nocc] = numpy.einsum('sp,qp->sq', Fc[:,ncore:nocc], casdm1())

        Lrq = lib.einsum('Lxy,ya->Lxa',self.cderi, moa)
        #ca, cb = self.mol.sph2spinor_coeff()
        #moa_alpha = numpy.dot(ca, moa)
        #moa_beta  = numpy.dot(cb, moa)
        #Lrq = lib.einsum('Lij,xi,ja->Lia', self.cderi, ca.T.conj(), moa_alpha) \
        #    + lib.einsum('Lij,xi,ja->Lia', self.cderi, cb.T.conj(), moa_beta)
        Lpq = lib.einsum('Lxa,xb->Lba', Lrq, moa.conj())
        Lrq = lib.einsum('Lxa,xy->Lya', Lrq, mo.conj())
        paaa = lib.einsum('Lxa,Lcd->xacd', Lrq, Lpq)
        Grad[:,ncore:nocc]+= lib.einsum('ruvw,tvuw->rt', paaa, casdm2())

        E = nuc_energy + 0.5*numpy.sum((hcore+Fc).diagonal()[:ncore]) 
        E += numpy.einsum('tu, tu', Fc[ncore:nocc, ncore:nocc], casdm1())
        E += 0.5*lib.einsum('ruvw,rvuw', paaa[ncore:nocc], casdm2())
        print("calc grad df", time.clock() - start[0], time.time()-start[1]) 
        return 2*Grad, E

    def calcEDF(self, mo, casdm1, casdm2):
        hcore = self.hcore #_scf.get_hcore()

        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact
        nuc_energy = self.energy_nuc()
 
        moc, moa = mo[:,:ncore], mo[:,ncore:nocc]
        j, k = get_jk_df(self.cderi, moc)

        hcore = reduce(numpy.dot, (mo.conj().T, hcore , mo))
        Fc =  (hcore + reduce(numpy.dot, (mo.conj().T, (j-k), mo)))
        E = nuc_energy + 0.5*numpy.sum((hcore+Fc).diagonal()[:ncore]) 
        E += numpy.einsum('tu, tu', Fc[ncore:nocc, ncore:nocc], casdm1())
 
        Lrq = lib.einsum('Lxy,ya->Lxa',self.cderi, moa)
        Lpq = lib.einsum('Lxa,xb->Lba', Lrq, moa.conj())
        Lrq = lib.einsum('Lxa,xy->Lya', Lrq, moa.conj())
        paaa = lib.einsum('Lxa,Lcd->xacd', Lrq, Lpq)
        E += 0.5*lib.einsum('ruvw,rvuw', paaa, casdm2())

        return E
    #@profile
    def calcEDF_scalar(self, mo, casdm1, casdm2):
        start = time.clock(), time.time()
        cderi = self.cderi_scalar

        hcore = self._scf.get_hcore()

        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact
        nuc_energy = self.energy_nuc()

        moc, moa = mo[:,:ncore], mo[:,ncore:nocc]
        j, k = get_jk_df_scalar(self.mol, cderi, moc)

        hcore = reduce(numpy.dot, (mo.conj().T, hcore , mo))
        Fc =  (hcore + reduce(numpy.dot, (mo.conj().T, (j-k), mo)))
        E = nuc_energy + 0.5*numpy.sum((hcore+Fc).diagonal()[:ncore]) 
        E += numpy.einsum('tu, tu', Fc[ncore:nocc, ncore:nocc], casdm1())

        ca, cb = self.mol.sph2spinor_coeff()
        moa_alpha = numpy.dot(ca, moa)
        moa_beta  = numpy.dot(cb, moa)
        #print(cderi.shape, moa.shape, ca.shape, ca.T.conj().shape)
        #cderi_spinor = lib.einsum('pxy,xi,yj->pij', cderi, ca.conj(), ca)
        Lrq = lib.einsum('Lxj,xi,ja->Lia', cderi, ca.conj(), moa_alpha) \
            + lib.einsum('Lxj,xi,ja->Lia', cderi, cb.conj(), moa_beta)
        Lpq = lib.einsum('Lxa,xb->Lba', Lrq, moa.conj())
        Lrq = lib.einsum('Lxa,xy->Lya', Lrq, moa.conj())
        paaa = lib.einsum('Lxa,Lcd->xacd', Lrq, Lpq)

        E += 0.5*lib.einsum('ruvw,rvuw', paaa, casdm2())
        #print("calc E df from scalar cderi", time.clock() - start[0], time.time()-start[1]) 
        return E
    #@profile
    def calcGrad(self, mo, casdm1, casdm2, ERIS=None):
        t0 = time.clock(), time.time()
        log = logger.Logger(self.stdout, verbose=self.verbose)
        if ERIS is None: ERIS = self.ao2mo(mo,level=2)
        hcore = self.hcore #_scf.get_hcore()
        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact

        nuc_energy = self.energy_nuc()
        Grad = numpy.zeros((nmo, nmo), dtype=complex)

        moc, moa = mo[:,:ncore], mo[:,ncore:nocc]
        dmcore = numpy.dot(moc, moc.T.conj())
        dmcas = reduce(numpy.dot, (moa.conj(), casdm1(), moa.T)).conj()
        j,k = self._scf.get_jk(self.mol, dm=dmcore)
        t0 = log.timer("core jk", *t0)     
        ja,ka = self._scf.get_jk(self.mol, dm=dmcas)
        t0 = log.timer("active jk", *t0)

        
        hcore = reduce(numpy.dot, (mo.conj().T, hcore , mo))
        #print (hcore.diagonal())
        Fc =  (hcore + reduce(numpy.dot, (mo.conj().T, (j-k), mo)))
        Grad[:,:ncore] = hcore[:,:ncore] + reduce(numpy.dot, (mo.conj().T, j + ja - k - ka, moc))
        t0=log.timer("some reducions", *t0)
        Grad[:,ncore:nocc] = lib.einsum('sp,qp->sq', Fc[:,ncore:nocc], casdm1())
        Grad[:,ncore:nocc]+= lib.einsum('ruvw,tvuw->rt', ERIS.paaa, casdm2())
        t0 = log.timer("paaa transformation", *t0)

        Ecore = 0.5*numpy.sum((hcore+Fc).diagonal()[:ncore]) 
        Ecas1 = lib.einsum('tu, tu', Fc[ncore:nocc, ncore:nocc], casdm1())
        Ecas2 = 0.5*lib.einsum('tuvw, tvuw', ERIS.paaa[ncore:nocc], casdm2())
        t0 = log.timer("reduce energy from dm", *t0)
        #print("time of calc grad:", time.clock()-start[0], time.time() - start[1])
        return 2*Grad, (Ecore+Ecas1+Ecas2+nuc_energy).real
    

    def calcGradOld(self, mo, casdm1, casdm2, ERIS=None):
        if ERIS is None: ERIS = self.ao2mo(mo,level=2)
        hcore = self.hcore #_scf.get_hcore()
        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact

        Grad = numpy.zeros((nmo, nmo), dtype=complex)

        moc, moa = mo[:,:ncore], mo[:,ncore:nocc]
        dmcore = numpy.dot(moc, moc.T.conj())
        dmcas = reduce(numpy.dot, (moa.conj(), casdm1(), moa.T)).conj()
        j,k = self._scf.get_jk(self.mol, dm=dmcore)        
        ja,ka = self._scf.get_jk(self.mol, dm=dmcas)        

        
        hcore = reduce(numpy.dot, (mo.conj().T, hcore , mo))
        Fc =  (hcore + reduce(numpy.dot, (mo.conj().T, (j-k), mo)))
        Grad[:,:ncore] = hcore[:,:ncore] + reduce(numpy.dot, (mo.conj().T, j + ja - k - ka, moc))
        
        Grad[:,ncore:nocc] = lib.einsum('sp,qp->sq', Fc[:,ncore:nocc], casdm1())
        Grad[:,ncore:nocc]+= lib.einsum('ruvw,tvuw->rt', ERIS.paaa, casdm2())
        return 2*Grad

    ###IT IS NOT CORRECT, maybe some day i will fix it###
    def calcH(self, mo, x, casdm1, casdm2, ERIS):
        hcore = self.hcore #_scf.get_hcore()
        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact

        moc, moa = mo[:,:ncore], mo[:,ncore:nocc]
        dmcore = numpy.dot(moc, moc.T.conj())
        dmcas = reduce(numpy.dot, (moa.conj(), casdm1(), moa.T)).conj()
        j,k = self._scf.get_jk(self.mol, dm=dmcore)        
        ja,ka = self._scf.get_jk(self.mol, dm=dmcas)        

        hcore = reduce(numpy.dot, (mo.conj().T, hcore , mo))
        Fc =  (hcore + reduce(numpy.dot, (mo.conj().T, (j-k), mo)))
        Fa = reduce(numpy.dot, (mo.conj().T, (ja-ka), mo))

        Hrr = numpy.zeros((nmo,nocc, nmo, nocc))
        for i in range(ncore):
            Hrr[:,i,:,i] = (Fc+Fa).real

        print (Hrr[2,2,3,3])
        Hrr[:,:ncore,:,:ncore] += \
            (numpy.einsum('xijy->xiyj',ERIS.poop[:,:ncore,:ncore])\
            - numpy.einsum('xyji->xiyj',ERIS.ppoo[:,:,:ncore,:ncore])\
            + numpy.einsum('xiyj->xiyj', ERIS.popo[:,:ncore,:,:ncore])\
            - numpy.einsum('xjyi->xiyj', ERIS.popo[:,:ncore,:,:ncore])).real

        Hrr[:,:ncore, :,ncore:nocc] +=\
            (numpy.einsum('xqyi,pq->yixp',ERIS.popo[:,ncore:nocc,:,:ncore], casdm1())\
            +numpy.einsum('xqiy,pq->yixp',ERIS.poop[:,ncore:nocc,:ncore,:], casdm1())\
            -numpy.einsum('xiyq,pq->yixp',ERIS.popo[:,:ncore,:,ncore:nocc], casdm1())\
            -numpy.einsum('yxpi,pq->yixq',ERIS.ppoo[:,:,ncore:nocc,:ncore], casdm1())).real

        Hrr[:,ncore:nocc,:,ncore:nocc] +=\
            (0*numpy.einsum('xy,pq->xpyq',Fc,casdm1())\
            + 1*numpy.einsum('xyrs,prqs->xpyq',ERIS.ppoo[:,:,ncore:nocc,ncore:nocc].real, casdm2())\
            + 0.5*numpy.einsum('xsry,rpqs->xpyq',ERIS.poop[:,ncore:nocc,ncore:nocc,:], casdm2())\
            + 0.5*numpy.einsum('xsry,prsq->xpyq',ERIS.poop[:,ncore:nocc,ncore:nocc,:], casdm2())\
            + 0.5*numpy.einsum('ysxr,pqrs->xpyq',ERIS.popo[:,ncore:nocc,:,ncore:nocc], casdm2())\
            + 0.5*numpy.einsum('yrxs,srpq->xpyq',ERIS.popo[:,ncore:nocc,:,ncore:nocc].conj(), casdm2())).real

        return 2*Hrr

    def calcE(self, mo, casdm1, casdm2, ERIS=None):
        if ERIS is None: ERIS = self.ao2mo(mo, level=1)
        hcore = self._scf.get_hcore()
        ncore, nact = self.ncore, self.ncas
        nocc = ncore+nact
        nuc_energy = self.energy_nuc()
        
        moc = mo[:,:ncore]
        dmcore = numpy.dot(moc, moc.conj().T)
        j,k = self._scf.get_jk(self.mol, dm=dmcore)        

        
        hcore = reduce(numpy.dot, (mo.conj().T, hcore , mo))
        Fc = ( hcore + reduce(numpy.dot, (mo.conj().T, (j-k), mo)))
        Ecore = 0.5*numpy.sum((hcore+Fc).diagonal()[:ncore]) 

        Ecas1 = numpy.einsum('tu, tu', Fc[ncore:nocc, ncore:nocc], casdm1())
        Ecas2 = 0.5*numpy.einsum('tuvw, tvuw', ERIS.aaaa, casdm2())

        return Ecore+Ecas1+Ecas2+nuc_energy


    def uniq_var_indices(self, nmo, ncore, ncas, frozen=None):
        nocc = ncore + ncas
        mask = numpy.zeros((nmo,nmo),dtype=bool)
        mask[ncore:nocc,:ncore] = True
        mask[nocc:,:nocc] = True
        if self.internal_rotation:
            mask[ncore:nocc,ncore:nocc][numpy.tril_indices(ncas,-1)] = True
        if frozen is not None:
            if isinstance(frozen, (int, numpy.integer)):
                mask[:frozen] = mask[:,:frozen] = False
            else:
                frozen = numpy.asarray(frozen)
                mask[frozen] = mask[:,frozen] = False
        return mask

    def pack_vars(self, mat):
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)

        vec1 = mat[idx].real
        vec2 = mat[idx].imag
        vec = numpy.zeros((2*vec1.shape[0],))
        vec[:vec1.shape[0]] = 1*vec1
        vec[vec1.shape[0]:] = 1*vec2
        return vec

    # to anti symmetric matrix
    def unpack_vars(self, v):
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        mat = numpy.zeros((nmo,nmo), dtype=complex)
        nvars = v.shape[0]//2
        mat[idx] += v[:nvars]
        mat[idx] += 1j*v[nvars:]
        return mat - mat.T.conj()

    #@profile
    def optimizeOrbs(self, mo, casdm1, casdm2, addnoise, eris, r0, conv_tol, log):
        #the part of mo that is relevant 
        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact
        cput0 = t2m = t1m = (time.clock(), time.time())
        def NewtonStep(casscf, mo, nocc, Grad):
            G = casscf.pack_vars(Grad-Grad.conj().T)
            #Grad0_old, eold = casscf.calcGradDF(mo, casdm1, casdm2)
            t0 = (time.clock(), time.time())
            Grad0, e = casscf.calcGradDFScalar(mo, casdm1, casdm2)
            #print(numpy.linalg.norm(Grad0), numpy.linalg.norm(Grad0_old), numpy.linalg.norm(Grad0-Grad0_old))
            #print(e-eold)
            G0 = casscf.pack_vars(Grad0-Grad0.conj().T)
            
            def hop(x):
                Gradnewp = 0.*mo
                x_unpack = casscf.unpack_vars(x)
                eps = 1.e-5
                Kappa = eps*x_unpack
                
                monew = numpy.dot(mo, expmat(Kappa))
                t0 = (time.clock(), time.time())
                Gradnewp, e = casscf.calcGradDFScalar(monew, casdm1, casdm2)
                #t0 = log.timer_debug1('df approximated gradient', *t0)
                #print (numpy.linalg.norm(Gradnewp-Grad))  

                f = numpy.dot(Kappa.conj().T, Gradnewp)
                h = numpy.dot(Gradnewp, Kappa.conj().T)
                Gradnewp = Gradnewp - 0.5*(f-h) #- Gtemp.conj().T - 0.5*(f-f.conj().T+h-h.conj().T)
                Gradnewp = Gradnewp - Gradnewp.conj().T
                Gnewp= casscf.pack_vars(Gradnewp)
              
                
                Hx = (Gnewp - G0)/eps
                #print ("hop")
                return Hx

            x0 = 0.*G
            x, norm = GMRES(hop, -G, x0)
            #x, stat = scipy.sparse.linalg.gmres(hop, -G, x0,maxiter = 10)
            return x, norm

        imicro, nmicro, T, Grad = 0, 5, numpy.zeros_like(mo), 0.*mo
        Enew = 0.
        #Eold = self.calcE(mo, casdm1, casdm2).real
        Grad, Eold = self.calcGrad(mo, casdm1, casdm2)
        t2m = t1m = log.timer('exact gradient', *cput0)
        #Eolddf = self.calcEDF(mo, casdm1, casdm2).real
        Eolddf = self.calcEDF_scalar(mo, casdm1, casdm2).real
        t2m = log.timer_debug1('energy approximated from dm integrals', *t2m)
        print("norm(g): %6.2g" % (numpy.linalg.norm(Grad-Grad.conj().T)))
        while True:
            tau = 1.0
            gnorm = numpy.linalg.norm(Grad-Grad.conj().T)

            #if gradient is converged then exit
            if ( gnorm < conv_tol or imicro >= nmicro or tau <= 1.e-2):      
                return mo, Grad-Grad.conj().T, imicro, gnorm
                

            #find the newton step direction
            x, gnorm = NewtonStep(self, mo, nocc, Grad)
            t1m = t2m = log.timer('one newton step costs', *t1m)
            T = self.unpack_vars(x)
 
            ###do line search along the AH direction
            while tau > 1e-2:
                monew = numpy.dot(mo, expmat(tau*(T) ))
                Enewdf = self.calcEDF_scalar(monew, casdm1, casdm2).real
                t2m = log.timer('energy approximated with df', *t2m)
                #print ("line search ", Enewdf, Eolddf)
                if (Enewdf < Eolddf or tau/2 <= 1.e-3):# - tau * 1e-4*gnorm):
                    Grad, Enew = self.calcGrad(monew, casdm1, casdm2)
                    t1m = t2m = log.timer('energy updated in micro iteration', *t1m)
                    print ("%d  %6.3e  %18.12g   %13.6e   g=%6.2e"\
                    %(imicro, tau, Enew, Enew-Eold, gnorm))
                    Eold = Enew
                    Eolddf = Enewdf
                    mo = 1.*monew
                    break
                tau = tau/2.    
            imicro += 1

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 4
    mol.memory=20000
    mol.output = None#"out_h2o"
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]

    #mol.basis = 'cc-pvtz'
    mol.basis = '6-31g'
    mol.build()

    '''
    m = scf.RHF(mol)
    ehf = m.kernel()
    print (ehf)
    mc = mcscf.CASSCF(m, 6,6)
    emc = mc.kernel()[0]
    print (emc)
    '''
    
    m = scf.X2C(mol)
    #m = scf.GHF(mol)
    ehf = m.kernel()
    print (ehf)
    #mc = ZCASSCF(m, 16, 8)
    mc = ZCASSCF(m, 8, 4)
    mc.fcisolver = shci.SHCI(mol)
    mc.fcisolver.sweep_epsilon=[1.e-5]
    mc.fcisolver.sweep_iter=[0]
    mc.fcisolver.davidsonTol = 1.e-6

    mo = 1.*m.mo_coeff
    
    numpy.random.seed(5)
    noise = numpy.zeros(mo.shape, dtype=complex)
    noise = numpy.random.random(mo.shape) +\
                numpy.random.random(mo.shape)*1.j
    mo = numpy.dot(mo, expmat(-0.01*(noise - noise.T.conj())))
    mc.kernel(mo)
    #import cProfile
    #cProfile.run('mc.kernel(mo)')
    #exit(0)

    '''
    emc = mc.kernel(mo)[0]
    exit(0)
    print(ehf, emc, emc-ehf)
    print(emc - -3.22013929407)
    exit(0)
    '''
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvtz',
                 'O': 'cc-pvtz',}
    mol.build()

    m = scf.DHF(mol)
    ehf = m.scf()

    from pyscf.df import density_fit
    m2 = density_fit(m, "cc-pvtz-jkfit")
    energy = m2.scf()
    print (ehf, energy)
    exit(0)

    mc = mc1step.CASSCF(m, 6, 4)
    mc.verbose = 5
    mo = m.mo_coeff.copy()
    mo[:,2:5] = m.mo_coeff[:,[4,2,3]]
    emc = mc.mc2step(mo)[0]
    print(ehf, emc, emc-ehf)
    #-76.0267656731 -76.0873922924 -0.0606266193028
    print(emc - -76.0873923174, emc - -76.0926176464)



            
 
