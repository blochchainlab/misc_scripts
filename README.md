# Miscellaneous small scripts

- Inspect bvec to compare the quality of the approximate uniform spread with and without antipodal symmetry  
- Fit DTI over multi-shell dataset to evaluate diffusivity for postmortem scan  

## Optimal b-value calculation helper scripts
These scripts work best on dataset with large number of different b-value  

**invMD_bmax.py**  
Produces helper maps to evaluate a choice of postmortem b-value  

- DTI WLS fit over the full bvec/bval dataset and a mask  
- returns various maps  
    - mean diffusivity: MD
    - MD<sup>-1</sup>
    - Fractional Anisotropy: FA
    - Inverse Largest eigenvalue: &lambda;<sub>max</sub><sup>-1</sup>
    - Maximum DTI signal contrast on last shell: 
    $$\Delta S = \exp(-b_{\max}\lambda_{\min}) - \exp(-b_{\max}\lambda_{\max})$$


**bvalue_estimator.py**  
Produces helper histograms to choose of postmortem b-value  

- fit DTI WLS using all bvecs/bvals up to some b<sub>max</sub> and iteratively increases b<sub>max</sub>
- returns various metrics based on the MD distributions  
    -  Histograms of MD for each b<sub>max</sub>  
    -  Graph of inverse histogram peaks as a function of b<sub>max</sub>  
    -  Graph of inverse 50%-quantile of histogram as a function of b<sub>max</sub>  


**deltaS_estimator.py**  
Produces helper histograms to choose of postmortem b-value  

- fit DTI WLS using all bvecs/bvals up to some b<sub>max</sub> and *iteratively increases b<sub>max</sub>*
- returns various metrics based on the distributions of Maximum DTI signal contrast on last shell 
$$\Delta S = \exp(-b_{\max}\lambda_{\min}) - \exp(-b_{\max}\lambda_{\max})$$  
    -  Histograms of \Delta S for each b<sub>max</sub>  
    -  Graph of inverse 25%- 50%- and 75%- quantile of histogram as a function of b<sub>max</sub>  

**deltaS_estimator.py**  
Produces helper histograms to choose of postmortem b-value  

- fit DTI WLS using *each b-value shell*
- returns various metrics based on the distributions of Maximum DTI signal contrast on shell 
$$\Delta S = \exp(-b\lambda_{\min}) - \exp(-b\lambda_{\max})$$  
    -  Histograms of \Delta S for each b-value  
    -  Graph of inverse 25%- 50%- and 75%- quantile of histogram as a function of b-value  






### TODO  
List each script with more details  
Give example call for each script  
List dependencies for each script (especially the rare-ish one like dipy or scilpy)

