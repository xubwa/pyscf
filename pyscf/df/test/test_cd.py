#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

import os
import tempfile
import unittest
import numpy
from scipy.spatial.transform import Rotation
from pyscf import gto, scf, lib, df, mp, cc


def setUpModule():
    global mol, eri
    mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='cc-pvdz',
                verbose=5, output='/dev/null')
    eri = mol.intor('int2e', aosym='s4')


def tearDownModule():
    global mol, eri
    mol.stdout.close()
    del mol, eri


def _build(m, method='twostep', tau=1e-5, **kwargs):
    c = df.CD(m, tau=tau, method=method, **kwargs)
    return c.build()._cderi


def _benzene(basis, rot=numpy.eye(3)):
    r, R = 1.396, 2.479
    ang = numpy.arange(6) * numpy.pi / 3
    atoms = ([('C', rot.dot((r*numpy.cos(a), r*numpy.sin(a), 0))) for a in ang]
             + [('H', rot.dot((R*numpy.cos(a), R*numpy.sin(a), 0))) for a in ang])
    return gto.M(atom=atoms, basis=basis, verbose=0)


class KnownValues(unittest.TestCase):
    def test_accuracy(self):
        # every assembled integral within tau
        for method in ('onestep', 'twostep'):
            for tau in (1e-4, 1e-6):
                L = _build(mol, method, tau)
                self.assertLessEqual(abs(eri - L.T.dot(L)).max(), tau)

    def test_onestep_twostep(self):
        # the same pivots, the same decomposition
        c1 = df.CD(mol, tau=1e-5, method='onestep').build()
        c2 = df.CD(mol, tau=1e-5, method='twostep').build()
        L1, L2 = c1._cderi, c2._cderi
        self.assertEqual(L1.shape, L2.shape)
        self.assertAlmostEqual(abs(L1.T.dot(L1) - L2.T.dot(L2)).max(), 0, 12)
        self.assertEqual(c2.metric_chol.shape, (L2.shape[0],) * 2)

    def test_scf(self):
        # the CD mean field is the RHF of the assembled integrals L^T L
        mf = scf.RHF(mol).density_fit(with_df=df.CD(mol, tau=1e-6))
        e_cd = mf.run(conv_tol=1e-11).e_tot
        L = mf.with_df._cderi
        ref = scf.RHF(mol)
        ref._eri = L.T.dot(L)
        self.assertAlmostEqual(e_cd, ref.run(conv_tol=1e-11).e_tot, 9)
        self.assertAlmostEqual(e_cd, -76.02676639601, 8)

    def test_ccsd(self):
        # post-HF methods that take the vectors from with_df reuse them: CCSD
        # of the CD mean field is the CCSD of the assembled integrals
        mf = scf.RHF(mol).density_fit(with_df=df.CD(mol, tau=1e-6)).run(conv_tol=1e-11)
        e_cd = cc.CCSD(mf).run(conv_tol=1e-10).e_corr
        L = mf.with_df._cderi
        ref = scf.RHF(mol)
        ref._eri = L.T.dot(L)
        ref.run(conv_tol=1e-11)
        self.assertAlmostEqual(cc.CCSD(ref).run(conv_tol=1e-10).e_corr, e_cd, 8)

    def test_mp2(self):
        # DF-MP2 reuses the vectors of the CD mean field
        mf = scf.RHF(mol).density_fit(with_df=df.CD(mol, tau=1e-6)).run(conv_tol=1e-11)
        e_cd = mp.MP2(mf).run().e_corr
        L = mf.with_df._cderi
        ref = scf.RHF(mol)
        ref._eri = L.T.dot(L)
        ref.run(conv_tol=1e-11)
        self.assertAlmostEqual(mp.MP2(ref).run().e_corr, e_cd, 10)

    def test_no_auxmol(self):
        # no atom-centred auxiliary basis: code that would build its own
        # auxiliary-basis integrals (e.g. DF-MP2 with an unbuilt with_df)
        # fails instead of silently ignoring the CD
        c = df.CD(mol, tau=1e-6)
        self.assertRaises(NotImplementedError, getattr, c, 'auxmol')
        self.assertRaises(NotImplementedError, getattr, c, 'auxbasis')
        mf = scf.RHF(mol).density_fit(with_df=c).run()
        pt = mp.MP2(mf)
        pt.with_df = df.CD(mol, tau=1e-6)       # not built
        self.assertRaises(NotImplementedError, pt.run)

    def test_rotation_invariance(self):
        rot = Rotation.from_euler('zyx', [0.7, 1.1, -0.4]).as_matrix()
        e = []
        for m in (_benzene('6-31g'), _benzene('6-31g', rot)):
            mf = scf.RHF(m).density_fit(with_df=df.CD(m, tau=1e-4))
            e.append(mf.run(conv_tol=1e-11).e_tot)
        self.assertAlmostEqual(e[1], e[0], 9)

    def test_point_group_symmetry(self):
        # the pivots are closed under the point group: the degeneracies of D6h
        # benzene (e1g HOMO, e2u LUMO, ...) are exact in the CD mean field
        m = _benzene('6-31g')
        mf = scf.RHF(m).density_fit(with_df=df.CD(m, tau=1e-4))
        e = mf.run(conv_tol=1e-11).mo_energy
        de = numpy.diff(e)
        pairs = numpy.where(de < 1e-5)[0]
        nocc = m.nelectron // 2
        self.assertIn(nocc - 2, pairs)
        self.assertLess(de[pairs].max(), 1e-10)

    def test_save_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            fn = os.path.join(tmp, 'cderi.h5')
            cd = df.CD(mol, tau=1e-5)
            cd._cderi_to_save = fn
            e1 = scf.RHF(mol).density_fit(with_df=cd).run().e_tot
            self.assertEqual(df.CD.cderi_shape(fn)[1], mol.nao*(mol.nao+1)//2)
            cd = df.CD(mol)
            cd._cderi = fn
            e2 = scf.RHF(mol).density_fit(with_df=cd).run().e_tot
            self.assertAlmostEqual(e2, e1, 10)

    def test_max_memory(self):
        for method in ('onestep', 'twostep'):
            c = df.CD(mol, tau=1e-8, method=method)
            c.max_memory = lib.current_memory()[0] + .2
            self.assertRaises(MemoryError, c.build)

    def test_not_supported(self):
        self.assertRaises(ValueError, df.CD(mol, method='threestep').build)
        c = df.CD(mol, tau=1e-4)
        dm = numpy.eye(mol.nao)
        self.assertRaises(NotImplementedError, c.get_jk, dm, omega=.3)


if __name__ == "__main__":
    print("Full Tests for Cholesky-decomposed ERIs")
    unittest.main()
