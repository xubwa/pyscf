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

'''
Four-component Dirac-Hartree-Fock in quadruple precision
=========================================================

Integrals (McMurchie-Davidson), dense linear algebra (gemm, complex
Hermitian Jacobi eigensolver) and the SCF iterations are carried out in
IEEE binary128 arithmetic (__float128, libquadmath).  The molecule, basis
set and nuclear model are taken from :class:`pyscf.gto.Mole`; the double
precision DHF of :mod:`pyscf.scf.dhf` provides the initial guess.

Simple usage::

    >>> from pyscf import gto, qdhf
    >>> mol = gto.M(atom='Ne', basis='cc-pvdz')
    >>> mf = qdhf.DHF(mol).run()
    >>> mf.e_tot_str
    '-128.6318158549160002820250474875605'

Requires mpmath and the libqdhf C extension (built with libquadmath).
'''

from pyscf.qdhf import qarray
from pyscf.qdhf import integrals
from pyscf.qdhf import dhf
from pyscf.qdhf.qarray import QArray
from pyscf.qdhf.dhf import DHF, RHF, GHF
