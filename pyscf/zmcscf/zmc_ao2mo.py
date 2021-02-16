import sys, tempfile, ctypes, time, numpy, h5py
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf.mcscf import mc_ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo import outcore
from . import r_outcore
from pyscf import ao2mo




# level = 1: aaaa
# level = 2: paaa
class _ERIS(object):
    def __init__(self, zcasscf, mo, method='outcore', level=1):
        mol = zcasscf.mol
        nao, nmo = mo.shape
        ncore = zcasscf.ncore
        ncas = zcasscf.ncas
        nocc = ncore+ncas


        mem_incore, mem_outcore, mem_basic = mc_ao2mo._mem_usage(ncore, ncas, nmo)
        mem_now = lib.current_memory()[0]
        eri = zcasscf._scf._eri
        moc, moa, moo = mo[:,:ncore], mo[:,ncore:nocc], mo[:,:nocc]
        if (method == 'incore' or mol.incore_anyway):
            raise NotImplementedError
            '''
            if eri is None:
                eri = mol.intor('int2e_spinor', aosym='s8')

            if level == 1:
                self.aaaa = ao2mo.kernel(eri, moa, intor="int2e_spinor")
            elif level == 2:   
                self.paaa = ao2mo.kernel(eri, (mo, moa, moa, moa), 
                                         intor="int2e_spinor")
            elif level == 3:
                self.ppoo = ao2mo.kernel(eri, (mo, moa, moa, moa), 
                                         intor="int2e_spinor")
                self.papa = ao2mo.kernel(eri, (mo, moa, moa, moa), 
                                         intor="int2e_spinor")
                self.paaa = ao2mo.kernel(eri, (mo, moa, moa, moa), 
                                         intor="int2e_spinor")
            '''

        else:
            import gc
            gc.collect()
            log = logger.Logger(zcasscf.stdout, zcasscf.verbose)
            self.feri = lib.H5TmpFile()
            max_memory = max(3000, zcasscf.max_memory*.9-mem_now)
            if max_memory < mem_basic:
                log.warn('Calculation needs %d MB memory, over CASSCF.max_memory (%d MB) limit',
                         (mem_basic+mem_now)/.9, zcasscf.max_memory)
            if level == 1:
                r_outcore.general(mol, (moa, moa, moa, moa), 
                                  self.feri, dataname='aaaa', 
                                  intor="int2e_sph")
                self.aaaa = self.feri['aaaa'][:,:].reshape((ncas, ncas, ncas, ncas))
            else:   
                r_outcore.general(mol, (mo, moa, moa, moa),               
                                  self.feri, dataname='paaa',
                                  intor="int2e_sph")
                self.paaa = self.feri['paaa'][:,:].reshape((nmo, ncas, ncas, ncas))

                    
        if (level == 1):
            self.aaaa.shape = (ncas, ncas, ncas, ncas)            
        else:
            self.paaa.shape = (nmo, ncas, ncas, ncas)
            self.aaaa = self.paaa[ncore:nocc]            
