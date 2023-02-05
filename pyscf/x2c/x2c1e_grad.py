#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Analytical nuclear gradients for X2C1E method

Ref.
JCP 135, 084114 (2011); DOI:10.1063/1.3624397
Note 1: Though the title of this paper is sfx2c, but changing the
W matrix to the spin dependent one will give the two-component gradient.
Note 2: Due to the spin-orbit coupling from X2CAMF method is an atom only term,
it will not contribute to the nuclear gradients, and can be used directly.
Note 3: This implementation uses the j-adapted spinor basis, i.e. can be used for SpinorX2CHelper directly,
for SpinOrbitX2CHelper, the function for class will translate the j-adapted spinor basis to the spin-orbital one.
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.x2c import x2c
from pyscf.x2c.sfx2c1e_grad import _get_r1

def hcore_grad_generator_spinor(x2cobj, mol=None):
    '''nuclear gradients of 2-component X2C1E core Hamiltonian
    '''
    if mol is None: mol = x2cobj.mol
    xmol, contr_coeff = x2cobj.get_xmol(mol)

    if x2cobj.basis is not None:
        s22 = xmol.intor_symmetric('int1e_ovlp_spinor')
        s21 = gto.intor_cross('int1e_ovlp_spinor', xmol, mol)
        contr_coeff = lib.cho_solve(s22, s21)

    get_h1_xmol = gen_hfw(xmol, x2cobj.approx)
    def hcore_deriv(atm_id):
        h1 = get_h1_xmol(atm_id)
        if contr_coeff is not None:
            h1 = lib.einsum('pi,xpq,qj->xij', contr_coeff, h1, contr_coeff)

        if isinstance(x2cobj, x2c.SpinOrbitX2CHelper):
            ca, cb = mol.sph2spinor_coeff()
            h1so = numpy.zeros((h1.shape))
            nao = h1so.shape[-1] // 2
            h1so[:,:nao,:nao] = lib.einsum('xpq,pi,qj->xij', h1, ca, ca.conj().T)
            h1so[:,nao:,nao:] = lib.einsum('xpq,pi,qj->xij', h1, cb, cb.conj().T)
            h1so[:,:nao,nao:] = lib.einsum('xpq,pi,qj->xij', h1, ca, cb.conj().T)
            h1so[:,nao:,:nao] = lib.einsum('xpq,pi,qj->xij', h1, cb, ca.conj().T)
            h1 = h1so
        return numpy.asarray(h1)
    return hcore_deriv


def hcore_grad_generator_spinorbital(x2cobj, mol=None):
    '''nuclear gradients of 2-component X2C1E core Hamiltonian
    '''
    if mol is None: mol = x2cobj.mol
    xmol, contr_coeff = x2cobj.get_xmol(mol)
    print(contr_coeff.shape)

    if x2cobj.basis is not None:
        s22 = xmol.intor_symmetric('int1e_ovlp_spinor')
        s21 = gto.intor_cross('int1e_ovlp_spinor', xmol, mol)
        contr_coeff = lib.cho_solve(s22, s21)

    get_h1_xmol = gen_hfw(xmol, x2cobj.approx)
    def hcore_deriv(atm_id):
        h1 = get_h1_xmol(atm_id)
        if contr_coeff is not None:
            contr_coeff_2c = x2c._block_diag(contr_coeff)
            print(h1.shape, contr_coeff_2c.shape)

            h1 = lib.einsum('pi,xpq,qj->xij', contr_coeff_2c, h1, contr_coeff_2c)

        ca, cb = mol.sph2spinor_coeff()
        ca = ca.T
        cb = cb.T
        print(ca.shape)
        h1so = numpy.zeros((h1.shape), dtype=complex)
        nao = h1so.shape[-1] // 2
        h1so[:,:nao,:nao] = lib.einsum('xpq,pi,qj->xij', h1, ca, ca.conj())
        h1so[:,nao:,nao:] = lib.einsum('xpq,pi,qj->xij', h1, cb, cb.conj())
        h1so[:,:nao,nao:] = lib.einsum('xpq,pi,qj->xij', h1, ca, cb.conj())
        h1so[:,nao:,:nao] = lib.einsum('xpq,pi,qj->xij', h1, cb, ca.conj())
        h1 = h1so
        return numpy.asarray(h1)
    return hcore_deriv

def gen_hfw(mol, approx='1E'):
    approx = approx.upper()
    c = lib.param.LIGHT_SPEED

    h0, s0 = _get_h0_s0(mol)
    e0, c0 = scipy.linalg.eigh(h0, s0)

    aoslices = mol.aoslice_2c_by_atom()
    nao = mol.nao_2c()

    if 'ATOM' in approx:
        x0 = numpy.zeros((nao,nao))
        for ia in range(mol.natm):
            ish0, ish1, p0, p1 = aoslices[ia]
            shls_slice = (ish0, ish1, ish0, ish1)
            t1 = mol.intor('int1e_kin_spinor', shls_slice=shls_slice)
            s1 = mol.intor('int1e_ovlp_spinor', shls_slice=shls_slice)
            with mol.with_rinv_at_nucleus(ia):
                z = -mol.atom_charge(ia)
                v1 = z * mol.intor('int1e_rinv_spinor', shls_slice=shls_slice)
                w1 = z * mol.intor('int1e_sprinvsp_spinor', shls_slice=shls_slice)
            x0[p0:p1,p0:p1] = x2c._x2c1e_xmatrix(t1, v1, w1, s1, c)
    else:
        cl0 = c0[:nao, nao:]
        cs0 = c0[nao:, nao:]
        x0 = scipy.linalg.solve(cl0.T, cs0.T).T

    s_nesc0 = s0[:nao, :nao] + reduce(numpy.dot, (x0.T, s0[nao:, nao:], x0))
    R0 = x2c._get_r(s0[:nao, :nao], s_nesc0)
    c_fw0 = numpy.vstack((R0, numpy.dot(x0, R0)))
    h0_fw_half = numpy.dot(h0, c_fw0)

    ovlp = mol.intor('int1e_ovlp_spinor')
    kin = mol.intor('int1e_kin_spinor')
    get_h1_etc = _gen_first_order_quantities(mol, e0, c0, x0, ovlp, kin, approx)

    def hcore_deriv(ia):
        h1_ao, s1_ao, e1, c1, x1, s_nesc1, R1, c_fw1 = get_h1_etc(ia)
        hfw1 = lib.einsum('xpi,pj->xij', c_fw1, h0_fw_half)
        hfw1 = hfw1 + hfw1.transpose(0,2,1)
        hfw1+= lib.einsum('pi,xpq,qj->xij', c_fw0, h1_ao, c_fw0)
        return hfw1
    return hcore_deriv

def _get_h0_s0(mol):
    c = lib.param.LIGHT_SPEED
    s = mol.intor_symmetric('int1e_ovlp_spinor')
    t = mol.intor_symmetric('int1e_kin_spinor')
    v = mol.intor_symmetric('int1e_nuc_spinor')
    w = mol.intor_symmetric('int1e_spnucsp_spinor')
    nao = s.shape[0]
    n2 = nao * 2
    dtype = numpy.result_type(t, v, w, s)
    h = numpy.zeros((n2,n2), dtype=dtype)
    m = numpy.zeros((n2,n2), dtype=dtype)
    h[:nao,:nao] = v
    h[:nao,nao:] = t
    h[nao:,:nao] = t
    h[nao:,nao:] = w * (.25/c**2) - t
    m[:nao,:nao] = s
    m[nao:,nao:] = t * (.5/c**2)
    return h, m

def _gen_h1_s1(mol):
    c = lib.param.LIGHT_SPEED
    s1 = mol.intor('int1e_ipovlp_spinor', comp=3)
    t1 = mol.intor('int1e_ipkin_spinor', comp=3)
    v1 = mol.intor('int1e_ipnuc_spinor', comp=3)
    w1 = mol.intor('int1e_ipspnucsp_spinor', comp=3)

    aoslices = mol.aoslice_2c_by_atom()
    nao = s1.shape[1]
    n2 = nao * 2

    def get_h1_s1(ia):
        h1 = numpy.zeros((3,n2,n2), dtype=v1.dtype)
        m1 = numpy.zeros((3,n2,n2), dtype=v1.dtype)
        ish0, ish1, i0, i1 = aoslices[ia]
        with mol.with_rinv_origin(mol.atom_coord(ia)):
            z = mol.atom_charge(ia)
            rinv1 = -z*mol.intor('int1e_iprinv_spinor', comp=3)
            sprinvsp1 = -z*mol.intor('int1e_ipsprinvsp_spinor', comp=3)
            rinv1 [:, i0:i1, :] -= v1[:, i0:i1]
            sprinvsp1[:, i0:i1, :] -= w1[:, i0:i1]

        for i in range(3):
            s1cc = numpy.zeros((nao, nao), dtype=complex)
            t1cc = numpy.zeros((nao, nao), dtype=complex)
            s1cc[:,i0:i1]-= s1[i,i0:i1].T
            t1cc[i0:i1,:] =-t1[i,i0:i1]
            t1cc[:,i0:i1]-= t1[i,i0:i1].T
            v1cc = rinv1[i]   + rinv1[i].T
            w1cc = sprinvsp1[i] + sprinvsp1[i].T

            h1[i,:nao,:nao] = v1cc
            h1[i,:nao,nao:] = t1cc
            h1[i,nao:,:nao] = t1cc
            h1[i,nao:,nao:] = w1cc * (.25/c**2) - t1cc
            m1[i,:nao,:nao] = s1cc
            m1[i,nao:,nao:] = t1cc * (.5/c**2)
        return h1, m1
    return get_h1_s1

def _gen_first_order_quantities(mol, e0, c0, x0, s0, t0, approx='1E'):
    c = lib.param.LIGHT_SPEED
    nao = e0.size // 2
    print(nao)
    n2 = nao * 2

    epq = e0[:,None] - e0
    degen_mask = abs(epq) < 1e-7
    epq[degen_mask] = 1e200

    cl0 = c0[:nao,nao:]
    # cs0 = c0[nao:,nao:]
    # s0 = mol.intor('int1e_ovlp')
    # t0 = mol.intor('int1e_kin')
    t0x0 = numpy.dot(t0, x0) * (.5/c**2)
    s_nesc0 = s0[:nao,:nao] + numpy.dot(x0.T, t0x0)

    w_s, v_s = scipy.linalg.eigh(s0)
    w_sqrt = numpy.sqrt(w_s)
    s_nesc0_vbas = reduce(numpy.dot, (v_s.T, s_nesc0, v_s))
    R0_mid = numpy.einsum('i,ij,j->ij', 1./w_sqrt, s_nesc0_vbas, 1./w_sqrt)
    wr0, vr0 = scipy.linalg.eigh(R0_mid)
    wr0_sqrt = numpy.sqrt(wr0)
    print(wr0)
    exit()
    # R0 in v_s basis
    R0 = numpy.dot(vr0/wr0_sqrt, vr0.T)
    R0 *= w_sqrt
    R0 /= w_sqrt[:,None]
    # Transform R0 back
    R0 = reduce(numpy.dot, (v_s, R0, v_s.T))

    get_h1_s1 = _gen_h1_s1(mol)
    def get_first_order(ia):
        h1ao, s1ao = get_h1_s1(ia)
        h1mo = lib.einsum('pi,xpq,qj->xij', c0.conj(), h1ao, c0)
        s1mo = lib.einsum('pi,xpq,qj->xij', c0.conj(), s1ao, c0)

        if 'ATOM' in approx:
            e1 = c1_ao = x1 = None
            s_nesc1 = lib.einsum('pi,xpq,qj->xij', x0, s1ao[:,nao:,nao:], x0)
            s_nesc1+= s1ao[:,:nao,:nao]
        else:
            f1 = h1mo[:,:,nao:] - s1mo[:,:,nao:] * e0[nao:]
            c1 = f1 / -epq[:,nao:]
            e1 = f1[:,nao:]
            e1[:,~degen_mask[nao:,nao:]] = 0

            c1_ao = lib.einsum('pq,xqi->xpi', c0, c1)
            cl1 = c1_ao[:,:nao]
            cs1 = c1_ao[:,nao:]
            tmp = cs1 - lib.einsum('pq,xqi->xpi', x0, cl1)
            x1 = scipy.linalg.solve(cl0.T, tmp.reshape(-1,nao).T)
            x1 = x1.T.reshape(3,nao,nao)

            s_nesc1 = lib.einsum('xpi,pj->xij', x1, t0x0)
            s_nesc1 = s_nesc1 + s_nesc1.transpose(0,2,1)
            s_nesc1+= lib.einsum('pi,xpq,qj->xij', x0, s1ao[:,nao:,nao:], x0)
            s_nesc1+= s1ao[:,:nao,:nao]

        R1 = numpy.empty((3,nao,nao), dtype=complex)
        c_fw1 = numpy.empty((3,n2,nao), dtype=complex)
        for i in range(3):
            R1[i] = _get_r1((w_sqrt,v_s), s_nesc0_vbas,
                            s1ao[i,:nao,:nao], s_nesc1[i], (wr0_sqrt,vr0))
            c_fw1[i,:nao] = R1[i]
            c_fw1[i,nao:] = numpy.dot(x0, R1[i])
            if 'ATOM' not in approx:
                c_fw1[i,nao:] += numpy.dot(x1[i], R0)
        return h1ao, s1ao, e1, c1_ao, x1, s_nesc1, R1, c_fw1
    return get_first_order
