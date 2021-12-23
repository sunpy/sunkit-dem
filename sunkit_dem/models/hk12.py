"""
Temporary port of Hannah and Kontar (2012) model.
See https://github.com/ianan/demreg/tree/master/python
"""
import numpy as np
from numpy import diag
from numpy.linalg import inv,pinv,svd
import distributed

from sunkit_dem.base_model import GenericModel

__all__ = ['HK12Model']


class HK12Model(GenericModel):
    
    def _model(self, alpha=1.0, increase_alpha=1.5, max_iterations=10, guess=None, use_em_loci=False, use_dask=False):
        errors = np.array([self.data[k].uncertainty.array.squeeze() for k in self._keys]).T
        dem, edem, elogt, chisq, dn_reg = dn2dem_pos(
            self.data_matrix.value.T,
            errors,
            self.kernel_matrix.value.T,
            np.log10(self.temperature_bin_centers.to(u.K).value),
            self.temperature_bin_edges.to(u.K).value,
            max_iter=max_iterations,
            reg_tweak=alpha,
            rgt_fact=increase_alpha,
            dem_norm0=guess,
            gloci=use_em_loci,
            use_dask=use_dask,
        )
        dem_unit = self.data_matrix.unit / self.kernel_matrix.unit / self.temperature_bin_edges.unit
        uncertainty = edem.T * dem_unit
        em = (dem * np.diff(self.temperature_bin_edges)).T * dem_unit
        dem = dem.T * dem_unit
        T_error_upper = self.temperature_bin_centers * (10**elogt -1 )
        T_error_lower = self.temperature_bin_centers * (1 - 1 / 10**elogt)
        return {'dem': dem,
                'uncertainty': uncertainty,
                'em': em,
                'temperature_errors_upper': T_error_upper,
                'temperature_errors_lower': T_error_lower,
                'chi_squared': np.atleast_1d(chisq)}
        
    @classmethod
    def defines_model_for(self, *args, **kwargs):
        return kwargs.get('model') == 'hk12'


def dn2dem_pos(dn_in,edn_in,tresp,tresp_logt,temps,
               reg_tweak=1.0, max_iter=10, gloci=0, rgt_fact=1.5, dem_norm0=None, use_dask=False):
    """
    Performs a Regularization on solar data, returning the Differential Emission Measure (DEM)
    using the method of Hannah & Kontar A&A 553 2013
    Basically getting DEM(T) out of g(f)=K(f,T)#DEM(T)

    --------------------
    Inputs:
    --------------------

    dn_in:
        The dn counts in dn/px/s for each filter is shape nx*ny*nf (nf=number of filters nx,ny = spatial dimensions, 
        one or both of which can be size 0 for 0d/1d problems. (or nt,nf or nx,nt,nf etc etc to get time series)
    edn_in:
        The error on the dn values in the same units and same dimensions.
    tresp:
        the temperature response matrix size n_tresp by nf
    tresp_logt:
        the temperatures in log t which the temperature response matrix corresponds to. E.G if your tresp matrix 
        runs from 5.0 to 8.0 in steps of 0.05 then this is the input to tresp_logt
    temps:
        the temperatures at which to calculate a DEM, array of length nt.

    --------------------
    Optional Inputs:
    --------------------

    dem_norm0:
        This is an array of length nt which contains an initial guess of the DEM solution providing a weighting 
        for the inversion process (L constraint matrix). The actual values of the normalisation 
        do not matter, only their relative values. 
        If no dem_norm0 given then L weighting based on value of gloci (0 is default)
    gloci:
        If no dem_norm0 given (or dem_norm0 array of 1s) then set gloci 1 or 0 (default 0) to choose weighting for the 
        inversion process (L constraint matrix).
        1: uses the min of EM loci curves to weight L.
        0: uses two reg runs - first with L=diag(1/dT) and DEM result from this used to weight L for second run. 
    reg_tweak:
        the initial normalised chisq target.
    max_iter:
        the maximum number of iterations to attempt, code iterates if negative DEM is produced. If max iter is reached before
        a suitable solution is found then the current solution is returned instead (which may contain negative values)
    rgt_fact:
        the factor by which rgt_tweak increases each iteration. As the target chisq increases there is more flexibility allowed 
        on the DEM

    --------------------
    Outputs:
    --------------------

    dem:
        The DEM, has shape nx*ny*nt and units cm^-5 K^-1
    edem:
        vertical errors on the DEM, same units.
    elogt:
        Horizontal errors on temperature.
    chisq:
        The final chisq, shape nx*ny. Pixels which have undergone more iterations will in general have higher chisq.
    dn_reg:
        The simulated dn counts, shape nx*ny*nf. This is obtained by multiplying the DEM(T) by the filter response K(f,T) for each channel
        useful for comparing with the initial data.
 
    """
    #create our bin averages:
    logt=([np.mean([(np.log10(temps[i])),np.log10((temps[i+1]))]) for i in np.arange(0,len(temps)-1)])
    #and widths
    dlogt=(np.log10(temps[1:])-np.log10(temps[:-1]))
    nt=len(dlogt)
    logt=(np.array([np.log10(temps[0])+(dlogt[i]*(float(i)+0.5)) for i in np.arange(nt)]))
    #number of DEM entries
    
    #hopefully we can deal with a variety of data, nx,ny,nf
    sze=dn_in.shape

    #for a single pixel
    if (np.any(dem_norm0)==None):
#         If no dem0 wght given just set them all to 1
        dem_norm0=np.ones(np.hstack((dn_in.shape[0:-1],nt)).astype(int))
    if len(sze)==1:
        nx=1
        ny=1
        nf=sze[0]
        dn=np.zeros([1,1,nf])
        dn[0,0,:]=dn_in
        edn=np.zeros([1,1,nf])
        edn[0,0,:]=edn_in
        if (np.all(dem_norm0) != None):
            dem0=np.zeros([1,1,nt])
            dem0[0,0,:]=dem_norm0
    #for a row of pixels
    if len(sze)==2:
        nx=sze[0]
        ny=1
        nf=sze[1]
        dn=np.zeros([nx,1,nf])
        dn[:,0,:]=dn_in
        edn=np.zeros([nx,1,nf])
        edn[:,0,:]=edn_in
        if (np.all(dem_norm0) != None):
            dem0=np.zeros([nx,1,nt])
            dem0[:,0,:]=dem_norm0
    #for 2d image
    if len(sze)==3:
        nx=sze[0]
        ny=sze[1]
        nf=sze[2]
        dn=np.zeros([nx,ny,nf])
        dn[:,:,:]=dn_in
        edn=np.zeros([nx,ny,nf])
        edn[:,:,:]=edn_in
        if (np.all(dem_norm0) != None):
            dem0=np.zeros([nx,ny,nt])
            dem0[:,:,:]=dem_norm0
    
    # Set glc to either none or all, based on gloci input (default none/not using)
# IDL version of code allows selective use of gloci, i.e [1,1,0,0,1,1] to chose 4 of 6 filters for EM loci
# dem_pix() in demmap_pos.py does allow this, but not sure will work through these wrapper functions
# also not sure if this functionality is actually needed, just stick with all filter or none?
    if gloci == 1:
        glc=np.ones(nf)
        glc.astype(int)
    else:
        glc=np.zeros(nf)
        glc.astype(int)      
        
    if len(tresp[0,:])!=nf:
        print('Tresp needs to be the same number of wavelengths/filters as the data.')
    
    truse=np.zeros([tresp[:,0].shape[0],nf])
    #check the tresp has no elements <0
    #replace any it finds with the mimimum tresp from the same filter
    for i in np.arange(0,nf):
        #keep good TR data
        truse[tresp[:,i] > 0]=tresp[tresp[:,i] > 0]
        #set bad data to the minimum
        truse[tresp[:,i] <= 0,i]=np.min(tresp[tresp[:,i] > 0],axis=0)[i]

    tr=np.zeros([nt,nf])
    for i in np.arange(nf):
#       Ideally should be interp in log-space, so changed
# Not as big an issue for purely AIA filters, but more of an issue for steeper X-ray ones
        tr[:,i]=10**np.interp(logt,tresp_logt,np.log10(truse[:,i]))
#     Previous version
#         tr[:,i]=np.interp(logt,tresp_logt,truse[:,i])

    rmatrix=np.zeros([nt,nf])
    #Put in the 1/K factor (remember doing it in logT not T hence the extra terms)
    for i in np.arange(nf):
        rmatrix[:,i]=tr[:,i]*10.0**logt*np.log(10.0**dlogt)
    #Just scale so not dealing with tiny numbers
    sclf=1E15
    rmatrix=rmatrix*sclf

    dn1d=np.reshape(dn,[nx*ny,nf])
    edn1d=np.reshape(edn,[nx*ny,nf])
#create our 1d arrays for output
    dem1d=np.zeros([nx*ny,nt])
    chisq1d=np.zeros([nx*ny])
    edem1d=np.zeros([nx*ny,nt])
    elogt1d=np.zeros([nx*ny,nt])
    dn_reg1d=np.zeros([nx*ny,nf])


# *****************************************************
#  Actually doing the DEM calculations
# *****************************************************
# Should always be just running the first part of if here as setting dem01d to array of 1s if nothing given
# So now more a check dimensions of things are correct 
    if ( dem0.ndim==dn.ndim ):
        dem01d=np.reshape(dem0,[nx*ny,nt])
        dem1d,edem1d,elogt1d,chisq1d,dn_reg1d=demmap_pos(dn1d,edn1d,rmatrix,logt,dlogt,glc,reg_tweak=reg_tweak,max_iter=max_iter,\
                rgt_fact=rgt_fact,dem_norm0=dem01d, use_dask=use_dask)
    else:
        dem1d,edem1d,elogt1d,chisq1d,dn_reg1d=demmap_pos(dn1d,edn1d,rmatrix,logt,\
            dlogt,glc,reg_tweak=reg_tweak,max_iter=max_iter,\
                rgt_fact=rgt_fact,dem_norm0=0, use_dask=use_dask)
    #reshape the 1d arrays to original dimensions and squeeze extra dimensions
    dem=((np.reshape(dem1d,[nx,ny,nt]))*sclf).squeeze()
    edem=((np.reshape(edem1d,[nx,ny,nt]))*sclf).squeeze()
    elogt=(np.reshape(elogt1d,[ny,nx,nt])/(2.0*np.sqrt(2.*np.log(2.)))).squeeze()
    chisq=(np.reshape(chisq1d,[nx,ny])).squeeze()
    dn_reg=(np.reshape(dn_reg1d,[nx,ny,nf])).squeeze()

    return dem,edem,elogt,chisq,dn_reg



def demmap_pos(dd,ed,rmatrix,logt,dlogt,glc,reg_tweak=1.0,max_iter=10,rgt_fact=1.5,dem_norm0=None, use_dask=False):
    """
    demmap_pos
    computes the dems for a 1 d array of length na with nf filters using the dn (g) counts and the temperature
    response matrix (K) for each filter.
    where 

        g=K.DEM

    Regularized approach solves this via
  
        ||K.DEM-g||^2 + lamb ||L.DEM||^2=min

    L is a zeroth order constraint matrix and lamb is the rrgularisation parameter

    The regularisation is solved via the GSVD of K and L (using dem_inv_gsvd)
    which provides the singular values (sva,svb) and the vectors u,v and w
    witht he properties U.T*K*W=sva*I and V.T L W = svb*I

    The dem is then obtained by:

        DEM_lamb = Sum_i (sva_i/(sva_i^2+svb_i^1*lamb)) * (g.u) w

    or

        K^-1=K^dag= Sum_i (sva_i/(sva_i^2+svb_i^1*lamb)) * u.w    

    We know all the bits of it apart from lamb. We get this from the Discrepancy principle (Morozon, 1967)
    such that the lamb chosen gives a DEM_lamb that produces a specified reduced chisq in data space which
    we call the "regularization parameter" (or reg_tweak) and we normally take this to be 1. As we also want a
    physically real solution (e.g. a DEM_lamb that is positive) we iteratively increase reg_tweak until a
    positive solution is found (or a max number of iterations is reached).

    Once a solution that satisfies this requirement is found the uncertainties are worked out:
    the vertical errors (on the DEM) are obtained by propagation of errors on dn through the
    solution; horizontal (T resolution) is how much K^dag#K deviates from I, so measuring
    spread from diagonal but also if regularization failed at that T.

    Inputs

    dd
        the dn counts for each channel
    ed
        the error on the dn counts
    rmatrix
        the trmatrix for each channel 
    logt
        log of the temperature bin averages
    dlogt
        size of the temperature bins
    glc
        indices of the filters for which gloci curves should be used to set the initial L constraint
        (if called from dn2dem_pos, then all 1s or 0s)

    Optional inputs

    reg_tweak
        initial chisq target
    rgt_fact
        scale factor for the increase in chi-sqaured target for each iteration
    max_iter
        maximum number of times to attempt the gsvd before giving up, returns the last attempt if max_iter reached
    dem_norm0
        provides a "guess" dem as a starting point, if none is supplied one is created.
    
    Outputs

    
    dem
        The DEM(T)
    edem
        the error on the DEM(T)
    elogt
        the error on logt    
    chisq
        the chisq for the dem compared to the dn
    dn_reg
        the simulated dn for each filter for the recovered DEM    
    """
 
    na=dd.shape[0]
    nf=rmatrix.shape[1]
    nt=logt.shape[0]
    #set up some arrays
    dem=np.zeros([na,nt])
    edem=np.zeros([na,nt])
    elogt=np.zeros([na,nt])
    chisq=np.zeros([na])
    dn_reg=np.zeros([na,nf])
 
    if use_dask:
        client = distributed.get_client()
        

        def run_dem_pix(i, data=None, errors=None, dem_norm=None, resp_matrix=None, log_temp=None, delta_log_temp=None):
            return dem_pix(data[i,:], errors[i,:], resp_matrix, log_temp, delta_log_temp, glc, 
                           reg_tweak=reg_tweak, max_iter=max_iter, rgt_fact=rgt_fact, dem_norm0=dem_norm[i,:])
        

        dd_scatter = client.scatter(dd)
        ed_scatter = client.scatter(ed)
        dem_norm_scatter = client.scatter(dem_norm0)
        rmatrix_scatter = client.scatter(rmatrix)
        logt_scatter = client.scatter(logt)
        dlogt_scatter = client.scatter(dlogt)
        futures = client.map(run_dem_pix, range(na),
                             data=dd_scatter, 
                             errors=ed_scatter, 
                             dem_norm=dem_norm_scatter,
                             resp_matrix=rmatrix_scatter,
                             log_temp=logt_scatter,
                             delta_log_temp=dlogt_scatter,
                             pure=False)
        results = client.gather(futures)
        for i,r in enumerate(results):
            dem[i,:] = r[0]
            edem[i,:] = r[1]
            elogt[i,:] = r[2]
            chisq[i] = r[3]
            dn_reg[i,:] = r[4]
    #else we execute in serial
    else:   
        for i in range(na):
            result=dem_pix(dd[i,:],ed[i,:],rmatrix,logt,dlogt,glc, \
                reg_tweak=reg_tweak,max_iter=max_iter,rgt_fact=rgt_fact,dem_norm0=dem_norm0[i,:])
            dem[i,:]=result[0]
            edem[i,:]=result[1]
            elogt[i,:]=result[2]
            chisq[i]=result[3]
            dn_reg[i,:]=result[4]

    return dem,edem,elogt,chisq,dn_reg


def dem_unwrap(dn,ed,rmatrix,logt,dlogt,glc,reg_tweak=1.0,max_iter=10,rgt_fact=1.5,dem_norm0=0):
    #this nasty function serialises the parallel blocks
    ndem=dn.shape[0]
    nt=logt.shape[0]
    nf=dn.shape[1]
    dem=np.zeros([ndem,nt])
    edem=np.zeros([ndem,nt])
    elogt=np.zeros([ndem,nt])
    chisq=np.zeros([ndem])
    dn_reg=np.zeros([ndem,nf])
    for i in range(ndem):
        result=dem_pix(dn[i,:],ed[i,:],rmatrix,logt,dlogt,glc, \
                reg_tweak=reg_tweak,max_iter=max_iter,rgt_fact=rgt_fact,dem_norm0=dem_norm0[i,:])
        dem[i,:]=result[0]
        edem[i,:]=result[1]
        elogt[i,:]=result[2]
        chisq[i]=result[3]
        dn_reg[i,:]=result[4]
    return dem,edem,elogt,chisq,dn_reg


def dem_pix(dnin,ednin,rmatrix,logt,dlogt,glc,reg_tweak=1.0,max_iter=10,rgt_fact=1.5,dem_norm0=0):
    
    nf=rmatrix.shape[1]
    nt=logt.shape[0]
    nmu=42
    ltt=min(logt)+1e-8+(max(logt)-min(logt))*np.arange(51)/(52-1.0)
    dem=np.zeros(nt)
    edem=np.zeros(nt)
    elogt=np.zeros(nt)
    chisq=0
    dn_reg=np.zeros(nf)

    rmatrixin=np.zeros([nt,nf])
    filt=np.zeros([nf,nt])  
    
    for kk in np.arange(nf):
        #response matrix
        rmatrixin[:,kk]=rmatrix[:,kk]/ednin[kk]
    dn=dnin/ednin
    edn=ednin/ednin
    # checking for Inf and NaN
    if ( sum(np.isnan(dn)) == 0 and sum(np.isinf(dn)) == 0 and np.prod(dn) > 0):
        ndem=1
        piter=0
        rgt=reg_tweak

        L=np.zeros([nt,nt])

        test_dem_reg=(np.zeros(1)).astype(int)
        
#  If you have supplied an initial guess/constraint normalized DEM then don't
#  need to calculate one (either from L=1/sqrt(dLogT) or min of EM loci)

# As the call to this now sets dem_norm to array of 1s if nothing provided by user can also test for that

#     Before calling this dem_norm0 is set to array of 1s if nothing provided by user
#     So we need to work out some weighting for L or is one provided as dem_norm0 (not 0 or array of 1s)?
        if (np.prod(dem_norm0) == 1.0 or dem_norm0[0] == 0):
# Need to work out a weighting here then, have two appraoches:
#         1. Do it via the min of em loci - chooses this if gloci, glc=1 from user
            if (np.sum(glc) > 0.0):
                gdglc=(glc>0).nonzero()[0]
                emloci=np.zeros((nt,gdglc.shape[0]))
                #for each gloci take the minimum and work out the emission measure
                for ee in np.arange(gdglc.shape[0]):
                    emloci[:,ee]=dnin[gdglc[ee]]/(rmatrix[:,gdglc[ee]])
                #for each temp we take the min of the loci curves as the estimate of the dem
                dem_model=np.zeros(nt)
                for ttt in np.arange(nt):
                    dem_model[ttt]=np.min(emloci[ttt,np.nonzero(emloci[ttt,:])])
                dem_reg_lwght=dem_model
#                ~~~~~~~~~~~~~~~~~ 
#             2. Or if nothing selected will run reg once, and use solution as weighting (self norm appraoch)
            else:
                # Calculate the initial constraint matrix
                # Just a diagional matrix scaled by dlogT
                L=diag(1.0/np.sqrt(dlogt[:]))
                #run gsvd
                sva,svb,U,V,W=dem_inv_gsvd(rmatrixin.T,L)
                #run reg map
                lamb=dem_reg_map(sva,svb,U,W,dn,edn,rgt,nmu)
                #filt, diagonal matrix
                for kk in np.arange(nf):
                    filt[kk,kk]=(sva[kk]/(sva[kk]**2+svb[kk]**2*lamb))
                kdag=W@(filt.T@U[:nf,:nf])
                dr0=(kdag@dn).squeeze()
                # only take the positive with certain amount (fcofmx) of max, then make rest small positive
                fcofmax=1e-4
                mask=np.where(dr0 > 0) and (dr0 > fcofmax*np.max(dr0))
                dem_reg_lwght=np.ones(nt)
                dem_reg_lwght[mask]=dr0[mask]
#                ~~~~~~~~~~~~~~~~~ 
#            Just smooth these inital dem_reg_lwght and max sure no value is too small
#             dem_reg_lwght=(np.convolve(dem_reg_lwght,np.ones(3)/3))[1:-1]/np.max(dem_reg_lwght[:])     
            dem_reg_lwght=(np.convolve(dem_reg_lwght[1:-1],np.ones(5)/5))[1:-1]/np.max(dem_reg_lwght[:])       
            dem_reg_lwght[dem_reg_lwght<=1e-8]=1e-8
        else:
#             Otherwise just set dem_reg to inputted weight   
            dem_reg_lwght=dem_norm0

    
#          Now actually do the dem regularisation using the L weighting from above
#  If set max_iter to 1 then wont have the pos constraint? As need following to run at least once
        while((ndem > 0) and (piter < max_iter)):
            #make L from 1/dem reg scaled by dlogt and diagonalise
            L=np.diag(np.sqrt(dlogt)/np.sqrt(abs(dem_reg_lwght))) 
            #call gsvd and reg map
            sva,svb,U,V,W = dem_inv_gsvd(rmatrixin.T,L)
            lamb=dem_reg_map(sva,svb,U,W,dn,edn,rgt,nmu)
            for kk in np.arange(nf):
                filt[kk,kk]=(sva[kk]/(sva[kk]**2+svb[kk]**2*lamb))
            kdag=W@(filt.T@U[:nf,:nf])
    
            dem_reg_out=(kdag@dn).squeeze()

            ndem=len(dem_reg_out[dem_reg_out < 0])
            rgt=rgt_fact*rgt
            piter+=1
      
        dem=dem_reg_out

        #work out the theoretical dn and compare to the input dn
        dn_reg=(rmatrix.T @ dem_reg_out).squeeze()
        residuals=(dnin-dn_reg)/ednin
        #work out the chisquared
        chisq=np.sum(residuals**2)/(nf)

        #do error calculations on dem
        delxi2=kdag@kdag.T
        edem=np.sqrt(np.diag(delxi2))

        kdagk=kdag@rmatrixin.T

        elogt=np.zeros(nt)
        for kk in np.arange(nt):
            rr=np.interp(ltt,logt,kdagk[:,kk])               
            hm_mask=(rr >= max(kdagk[:,kk])/2.)
            elogt[kk]=dlogt[kk]
            if (np.sum(hm_mask) > 0):
                elogt[kk]=(ltt[hm_mask][-1]-ltt[hm_mask][0])/2
    return dem,edem,elogt,chisq,dn_reg


def dem_reg_map(sigmaa,sigmab,U,W,data,err,reg_tweak,nmu=500):
    """
    dem_reg_map
    computes the regularisation parameter
    
    Inputs

    sigmaa: 
        gsv vector
    sigmab: 
        gsv vector
    U:      
        gsvd matrix
    V:      
        gsvd matrix
    data:   
        dn data
    err:    
        dn error
    reg_tweak: 
        how much to adjust the chisq each iteration

    Outputs

    opt:
        regularization paramater

    """
 

    nf=data.shape[0]
    nreg=sigmaa.shape[0]

    arg=np.zeros([nreg,nmu])
    discr=np.zeros([nmu])

    sigs=sigmaa[:nf]/sigmab[:nf]
    maxx=max(sigs)
    minx=min(sigs)**2.0*1E-2

    step=(np.log(maxx)-np.log(minx))/(nmu-1.)
    mu=np.exp(np.arange(nmu)*step)*minx
    for kk in np.arange(nf):
        coef=data@U[kk,:]-sigmaa[kk]
        for ii in np.arange(nmu):
            arg[kk,ii]=(mu[ii]*sigmab[kk]**2*coef/(sigmaa[kk]**2+mu[ii]*sigmab[kk]**2))**2
    
    discr=np.sum(arg,axis=0)-np.sum(err**2)*reg_tweak
  
    opt=mu[np.argmin(np.abs(discr))]

    return opt


def dem_inv_gsvd(A,B):
    """
    dem_inv_gsvd

    Performs the generalised singular value decomposition of two matrices A,B.

    Inputs

    A:
        cross section matrix
    B:
        regularisation matrix (square)

    Performs

    the decomposition of:

        A=U*SA*W^-1
        B=V*SB*W^-1

        with gsvd matrices u,v and the weight W and diagnoal matrics SA and SB

    Outputs

    U:
        decomposition product matrix
    V:
        decomposition prodyct matrix
    W:
        decomposition prodyct matrix
    alpha:
        the vector of the diagonal values of SA
    beta:
        the vector of the diagonal values of SB
  

    """  
    #calculate the matrix A*B^-1
    AB1=A@inv(B)
    sze=AB1.shape
    C=np.zeros([max(sze),max(sze)])
    C[:sze[0],:sze[1]]=AB1
    #use np.linalg.svd to calculate the singular value decomposition
    u,s,v = svd(C,full_matrices=True,compute_uv=True)
    # U, S, Vh = svd(AB1, full_matrices=False)
    #from the svd products calculate the diagonal components form the gsvd
    beta=1./np.sqrt(1+s**2)
    alpha=s*beta

    #diagonalise alpha and beta into SA and SB
    onea=np.diag(alpha)
    oneb=np.diag(beta)
     #calculate the w matrix
    # w=inv(inv(onea)@transpose(u)@A)
    w2=pinv(inv(oneb)@v@B)

    #return gsvd products, transposing v as we do.
    return alpha,beta,u.T[:,:sze[0]],v.T,w2
