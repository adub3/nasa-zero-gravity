<<<<<<< HEAD
# nasa-zero-gravity
=======
# nasa-zero-gravity

NASA API project.

## Inspiration
Recent climate disasters, especially wildfires in California and increasing drought severity, highlighted the need for predictive tools that can help communities prepare for natural disasters before they strike.
## What it does
Maps wildfire and drought risk in an interactive format. The data being displayed is simulated.
## How we built it
We developed a multi-source feature pipeline (not connected to the UI) that integrates Sentinel-2 optical imagery, Copernicus DEM data, ESA WorldCover land classification, and ERA5-Land meteorological data. Our 3D UNet model processes spatiotemporal data tensors [T, C, H, W] with multi-task learning for both spatial segmentation and disaster classification. The frontend uses React with Google Maps 3D API for visualization and user interaction.
## Challenges we ran into
Time was ultimately the biggest challenge. The vastness of our datasets. Model runtimes and errors.
## Accomplishments that we're proud of
Successfully integrated real satellite data from multiple sources using STAC protocols, implemented a working 3D UNet architecture for spatiotemporal prediction, created an intuitive 3D visualization interface, and built a no-args training system that simplifies model deployment. Implemented google-maps api to display model results. UI is clean and usable.
## What we learned
Working with cloud-native geospatial data at scale, applying deep learning to Earth observation problems, the complexity of multi-temporal satellite data processing, and not to bite off more than we can chew.
## What's next for Wildfire/Drought Prediction
Improve model accuracy through ensemble methods, add more disaster types, and pipeline to the frontend for live display.
