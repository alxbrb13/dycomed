###########
# Part 1.0  : get in situ data (optional in you already have some)
#############

Download in situ data using the Copernicus Marine Toolbox (https://pypi.org/project/copernicusmarine/). The toolbox should be installed and you should be logged in :

```
copernicusmarine login

python get_some_insitu_data.py
```
The above script uses the `get` function to collect vertical profiles. Interpolation on a common vertical grid is then performed to end up with a regular 2D array. It is fit to handle glider, CTD, Argo and XBT (the latter being sometimes very noisy). 

###########
#  Part 1.1 : Update existing data
############

Most of the time we have in situ data, but we want to update them. Or we gather in situ data from different datasets or providers and we want to merge them. The script below merge 2 datasets considering each data as point, regardeless they are points or vertical profiles.

```
python update_data.py
python merge_netcdf.py
```
############
# Part 2 : Prepare eddy detection data
##############
Available atlas with detection already performed:
- Dyned (AMEDA for Med Sea)
- META (Py Eddy Tracker at global scale)

Algorithms to *build* your own atlas :
- Py Eddy Tracker (Python) : https://github.com/AntSimi/py-eddy-tracker
- AMEDA (Matlab) : https://github.com/briaclevu/AMEDA

Eddy detections can come from various algorithm, you just need to provide detections per day (time in julian days since 2000-1-1 and as integers). 1 eddy observed 1 day constitutes 1 observation. The same eddy observed several days is 1 track. The code handles the case where no eddy tracking is provided.

1 eddy observation should provide at least 1 center and 1 surronding contour. Most of the algorithms compute 2 contours per observation, the 'maximal speed' and 'effective' contours, but the distinction is not necessary.

The only constraint is that contour coordinates ('x_max','y_max','x_eff','y_eff') MUST be have the same number of points, and should be 2D ( [#obs,#point_in_the_contour])


###########
# Part 3 : Perform colocation eddies <-> In situ
#########

In situ data are classified following the method explained in Barboni et al (2023) : https://doi.org/10.5194/os-19-229-2023  (see in particular Fig.2)

A time window of +/- 2 days (5 days : D-2,D-1,D0,D+1,D+2 with D0 cast date of the considered in situ data) is considered by default. An in situ data is labelled as 'inside-eddy' if it falls inside the maximal speed radius of an eddy at least 4 out of 5 days. It is labelled as 'outside-eddy' if it does not fall inside *any* contour in the time window. In situ data meeting none of the above criteria are labelled as 'ambiguous'. Expected collocation stats are roughly 10-20 % ambiguous, 10% inside-anticyclone, 10% inside-cyclone.

This 'eddy_lag' of +/- 2 days is the typical time variation of a noisy eddy detection, and can be extended or reduced. Very noisy detections should be checked with higher eddy_lag to ascertain collocation. On the other hand region where eddies movement is due to physical processes (for instance the to Beta drift) should have a lower 'eddy_lag'.


```
python eddy_link.py
```

########
# Part 4 : Compute background
########

Once in situ data are classified between 'inside-eddy' and 'outside-eddy', a reference background can be computed for each profile. This reference background is not a climatology but the mean of 'outside-eddy' profiles close in time and space to the considered profile.

Close in time and space should be tuned depending of the variable of interest : in-depth property, mixed-layer, atmospheric variable, etc.

```
python background.py
```

