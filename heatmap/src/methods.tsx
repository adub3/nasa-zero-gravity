import React from 'react';
import NavigationBar from './navigation-bar';

const MethodsPage = () => {
  return (
    <div style={{ 
      minHeight: '100vh', 
      background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
      fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
    }}>
      <NavigationBar />
      
      <div style={{
        paddingTop: '80px',
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '80px 2rem 2rem',
        lineHeight: '1.6'
      }}>
        
        <header style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <h1 style={{
            fontSize: '3rem',
            fontWeight: '700',
            color: '#333',
            marginBottom: '1rem'
          }}>
            Methodology
          </h1>
          <p style={{
            fontSize: '1.2rem',
            color: '#666',
            maxWidth: '800px',
            margin: '0 auto'
          }}>
            Our approach to real-time disaster risk prediction using satellite data and machine learning
          </p>
        </header>

        <div style={{
          background: 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(10px)',
          borderRadius: '16px',
          padding: '3rem',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          marginBottom: '3rem',
          textAlign: 'center'
        }}>
          <h2 style={{ color: '#d32f2f', marginBottom: '2rem', fontSize: '2.5rem' }}>Core Approach</h2>
          <p style={{ 
            color: '#555', 
            fontSize: '1.3rem', 
            lineHeight: '1.7',
            maxWidth: '800px',
            margin: '0 auto'
          }}>
            We developed a multi-temporal, multi-source feature pipeline using a 3D UNet architecture that processes 
            geospatial satellite data chips across multiple time periods. Our system integrates optical imagery, 
            meteorological data, topographical information, and land cover data to predict drought and wildfire risks 
            with spatial precision using advanced deep learning techniques.
          </p>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
          gap: '2rem',
          marginBottom: '3rem'
        }}>
          
          <div style={{
            background: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(10px)',
            borderRadius: '16px',
            padding: '2rem',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
            border: '1px solid rgba(255, 255, 255, 0.2)'
          }}>
            <h2 style={{ color: '#d32f2f', marginBottom: '1rem' }}>Data Sources</h2>
            <ul style={{ color: '#555', paddingLeft: '1.2rem' }}>
              <li><strong>Sentinel-2 L2A Optical Imagery:</strong> High-resolution multispectral satellite data (10m resolution) including bands B02, B03, B04, B08, B11, B12 with cloud masking via Scene Classification Layer (SCL)</li>
              <li><strong>Copernicus DEM:</strong> Digital elevation model with derived slope and aspect calculations for topographical analysis</li>
              <li><strong>ESA WorldCover:</strong> Global land cover classification data for vegetation and land use context</li>
              <li><strong>ERA5-Land Reanalysis:</strong> High-resolution meteorological data including precipitation, temperature, and atmospheric moisture</li>
              <li><strong>Derived Spectral Indices:</strong> NDVI (vegetation health), NDWI (water content), NDMI (moisture proxy), VPD (vapor pressure deficit)</li>
            </ul>
          </div>

          <div style={{
            background: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(10px)',
            borderRadius: '16px',
            padding: '2rem',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
            border: '1px solid rgba(255, 255, 255, 0.2)'
          }}>
            <h2 style={{ color: '#d32f2f', marginBottom: '1rem' }}>Machine Learning Architecture</h2>
            <ul style={{ color: '#555', paddingLeft: '1.2rem' }}>
              <li><strong>3D UNet Architecture:</strong> Deep neural network designed for spatiotemporal analysis with multi-temporal input processing [T, C, H, W] tensor format</li>
              <li><strong>Multi-Task Learning:</strong> Combined spatial segmentation head and dual classification heads for disaster type and subtype prediction</li>
              <li><strong>Geospatial Data Pipeline:</strong> Automated UTM projection and grid alignment for consistent spatial processing across different data sources</li>
              <li><strong>Temporal Integration:</strong> Multi-date lookback system incorporating historical patterns and temporal feature evolution</li>
              <li><strong>Cloud-Native Processing:</strong> STAC (SpatioTemporal Asset Catalog) integration with Microsoft Planetary Computer for scalable data access</li>
            </ul>
          </div>

        </div>

        <div style={{ marginBottom: '4rem' }}>
          <h2 style={{ 
            textAlign: 'center', 
            color: '#333', 
            marginBottom: '3rem',
            fontSize: '2.5rem'
          }}>
            Disaster-Specific Models
          </h2>
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
            gap: '2rem'
          }}>
            
            <div style={{
              background: 'rgba(255, 69, 0, 0.05)',
              backdropFilter: 'blur(10px)',
              borderRadius: '16px',
              padding: '2rem',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
              border: '2px solid rgba(255, 69, 0, 0.2)'
            }}>
              <h3 style={{ color: '#FF4500', marginBottom: '1.5rem', fontSize: '1.8rem' }}>🔥 Wildfire Risk Prediction</h3>
              <p style={{ color: '#555', marginBottom: '1rem', fontWeight: '600' }}>Key Indicators Analyzed:</p>
              <ul style={{ color: '#555', paddingLeft: '1.2rem', marginBottom: '1rem' }}>
                <li><strong>NDVI (Vegetation Index):</strong> (NIR - Red)/(NIR + Red) to assess vegetation health and dryness, typical range -0.2 to 0.9</li>
                <li><strong>NDMI (Moisture Index):</strong> (NIR - SWIR)/(NIR + SWIR) for vegetation moisture content analysis</li>
                <li><strong>Fire Weather Index (FWI):</strong> Simplified FWI calculation from ERA5-Land meteorological data</li>
                <li><strong>Vapor Pressure Deficit (VPD):</strong> Atmospheric dryness indicator from temperature and humidity</li>
                <li><strong>Topographical Factors:</strong> Slope and aspect derived from Copernicus DEM affecting fire spread patterns</li>
                <li><strong>Land Cover Classification:</strong> ESA WorldCover data to identify fire-prone vegetation types</li>
              </ul>
              <p style={{ color: '#666', fontSize: '0.9rem', fontStyle: 'italic' }}>
                Our 3D UNet architecture processes multi-temporal Sentinel-2 imagery combined with meteorological 
                and topographical data to predict wildfire ignition probability and spread risk patterns with 10-meter spatial resolution.
              </p>
            </div>

            <div style={{
              background: 'rgba(139, 69, 19, 0.05)',
              backdropFilter: 'blur(10px)',
              borderRadius: '16px',
              padding: '2rem',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
              border: '2px solid rgba(139, 69, 19, 0.2)'
            }}>
              <h3 style={{ color: '#8B4513', marginBottom: '1.5rem', fontSize: '1.8rem' }}>🌵 Drought Risk Assessment</h3>
              <p style={{ color: '#555', marginBottom: '1rem', fontWeight: '600' }}>Key Indicators Analyzed:</p>
              <ul style={{ color: '#555', paddingLeft: '1.2rem', marginBottom: '1rem' }}>
                <li><strong>NDWI (Water Index):</strong> (Green - NIR)/(Green + NIR) for vegetation water stress detection</li>
                <li><strong>Precipitation Patterns:</strong> ERA5-Land historical and current precipitation data aggregates</li>
                <li><strong>KBDI (Keetch-Byram Drought Index):</strong> Simplified drought severity calculation from meteorological variables</li>
                <li><strong>Vegetation Anomalies:</strong> Multi-temporal NDVI analysis to detect vegetation stress patterns</li>
                <li><strong>Surface Soil Moisture:</strong> SMAP integration framework (extensible for future implementation)</li>
                <li><strong>Temperature Anomalies:</strong> Surface temperature analysis from satellite thermal bands</li>
              </ul>
              <p style={{ color: '#666', fontSize: '0.9rem', fontStyle: 'italic' }}>
                Our drought model integrates spectral vegetation indices with meteorological time series to identify 
                early drought onset and severity, providing spatially-explicit drought risk maps at landscape scale.
              </p>
            </div>

          </div>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
          gap: '2rem',
          marginBottom: '3rem'
        }}>
          
          <div style={{
            background: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(10px)',
            borderRadius: '16px',
            padding: '2rem',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
            border: '1px solid rgba(255, 255, 255, 0.2)'
          }}>
            <h3 style={{ color: '#d32f2f', marginBottom: '1.5rem', fontSize: '1.5rem' }}>Our Challenge</h3>
            <p style={{ color: '#555', marginBottom: '1rem' }}>
              The CDC Hackathon challenged us to create innovative solutions for natural disaster preparedness. 
              We recognized that natural disasters pose significant health risks to communities and wanted to leverage
              satellite data to predict them with high spatial and temporal resolution.
            </p>
            <p style={{ color: '#555' }}>
              Our team decided to focus on predictive modeling using multi-temporal satellite imagery, combining 
              optical data with meteorological and topographical information for comprehensive risk assessment.
            </p>
          </div>

          <div style={{
            background: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(10px)',
            borderRadius: '16px',
            padding: '2rem',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
            border: '1px solid rgba(255, 255, 255, 0.2)'
          }}>
            <h3 style={{ color: '#d32f2f', marginBottom: '1.5rem', fontSize: '1.5rem' }}>What We Learned</h3>
            <ul style={{ color: '#555', paddingLeft: '1.2rem', listStyle: 'none' }}>
              <li style={{ marginBottom: '0.8rem', position: 'relative', paddingLeft: '1.5rem' }}>
                <span style={{ position: 'absolute', left: 0, color: '#d32f2f', fontWeight: 'bold' }}>•</span>
                <strong>Satellite Data Integration:</strong> Working with multi-source geospatial data at scale using STAC protocols
              </li>
              <li style={{ marginBottom: '0.8rem', position: 'relative', paddingLeft: '1.5rem' }}>
                <span style={{ position: 'absolute', left: 0, color: '#d32f2f', fontWeight: 'bold' }}>•</span>
                <strong>Deep Learning for Earth Observation:</strong> Applying 3D UNet architectures to spatiotemporal satellite data
              </li>
              <li style={{ marginBottom: '0.8rem', position: 'relative', paddingLeft: '1.5rem' }}>
                <span style={{ position: 'absolute', left: 0, color: '#d32f2f', fontWeight: 'bold' }}>•</span>
                <strong>Geospatial Processing:</strong> UTM reprojection and multi-temporal data alignment challenges
              </li>
              <li style={{ marginBottom: '0.8rem', position: 'relative', paddingLeft: '1.5rem' }}>
                <span style={{ position: 'absolute', left: 0, color: '#d32f2f', fontWeight: 'bold' }}>•</span>
                <strong>Impact:</strong> How advanced ML can enhance disaster preparedness and public health
              </li>
            </ul>
          </div>

        </div>

        <div style={{
          background: 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(10px)',
          borderRadius: '16px',
          padding: '2rem',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          marginBottom: '2rem'
        }}>
          <h2 style={{ color: '#d32f2f', marginBottom: '1.5rem' }}>Technical Implementation</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', alignItems: 'start' }}>
            <div>
              <h3 style={{ color: '#333', marginBottom: '1rem' }}>Geospatial Data Pipeline</h3>
              <p style={{ color: '#555', marginBottom: '1rem' }}>
                Our processing pipeline handles multi-source satellite data through automated workflows:
              </p>
              <ul style={{ color: '#555', paddingLeft: '1.2rem', marginBottom: '1rem' }}>
                <li><strong>UTM Reprojection:</strong> Automatic EPSG code selection (326** N, 327** S) based on coordinates</li>
                <li><strong>Grid Alignment:</strong> Common spatial grid (H=W=chip_size_m/resolution_m) for all data sources</li>
                <li><strong>Multi-temporal Stacking:</strong> Automated lookback date processing with cloud filtering (&lt; max_cloud_pct)</li>
                <li><strong>Scene Classification:</strong> SCL mask filtering keeping pixels with codes (2,4,5,6,7) representing valid land surfaces</li>
              </ul>
            </div>
            
            <div>
              <h3 style={{ color: '#333', marginBottom: '1rem' }}>Model Architecture & Training</h3>
              <p style={{ color: '#555', marginBottom: '1rem' }}>
                Our 3D UNet implementation processes spatiotemporal data with multiple output heads:
              </p>
              <ul style={{ color: '#555', paddingLeft: '1.2rem' }}>
                <li><strong>Input Format:</strong> [T, C, H, W] tensor with T=temporal lookbacks, C=channels</li>
                <li><strong>Spatial Segmentation:</strong> [1, T, H, W] output for pixel-level risk prediction</li>
                <li><strong>Multi-task Classification:</strong> Disaster type and subtype prediction heads</li>
                <li><strong>PyTorch Framework:</strong> scikit-learn integration with automated training loops</li>
              </ul>
            </div>
          </div>
          
          <div style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: 'rgba(211, 47, 47, 0.05)', borderRadius: '8px', border: '1px solid rgba(211, 47, 47, 0.1)' }}>
            <h3 style={{ color: '#d32f2f', marginBottom: '1rem' }}>Data Processing Modules</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
              <div style={{ fontSize: '0.9rem', color: '#555' }}>
                <strong>imagesample.py:</strong> Data fetching and processing orchestration
              </div>
              <div style={{ fontSize: '0.9rem', color: '#555' }}>
                <strong>nn.py:</strong> Model architecture, training loop, and no-args runner
              </div>
              <div style={{ fontSize: '0.9rem', color: '#555' }}>
                <strong>diagnostic.py:</strong> Connectivity tests and stackstac validation
              </div>
            </div>
          </div>
        </div>

        <div style={{
          background: 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(10px)',
          borderRadius: '16px',
          padding: '2rem',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          border: '1px solid rgba(255, 255, 255, 0.2)'
        }}>
          <h2 style={{ color: '#d32f2f', marginBottom: '1.5rem' }}>Implementation & Deployment</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '2rem' }}>
            <div>
              <h3 style={{ color: '#333', marginBottom: '1rem' }}>Development Environment</h3>
              <ul style={{ color: '#555', paddingLeft: '1.2rem' }}>
                <li><strong>Python 3.9+</strong> with PyTorch framework</li>
                <li><strong>Microsoft Planetary Computer</strong> for cloud-native data access</li>
                <li><strong>STAC Integration:</strong> Standardized geospatial data catalogs</li>
                <li><strong>No-args Training:</strong> Simplified execution without command line complexity</li>
              </ul>
            </div>
            
            <div>
              <h3 style={{ color: '#333', marginBottom: '1rem' }}>Data Processing</h3>
              <ul style={{ color: '#555', paddingLeft: '1.2rem' }}>
                <li><strong>Stackstac:</strong> Efficient raster data cube creation</li>
                <li><strong>Xarray:</strong> Multi-dimensional array processing</li>
                <li><strong>Automated UTM projection</strong> for consistent spatial alignment</li>
                <li><strong>Multi-temporal compositing</strong> with cloud filtering</li>
              </ul>
            </div>
            
            <div>
              <h3 style={{ color: '#333', marginBottom: '1rem' }}>Visualization Interface</h3>
              <ul style={{ color: '#555', paddingLeft: '1.2rem' }}>
                <li><strong>3D Globe Rendering:</strong> WebGL-powered Google Maps 3D API</li>
                <li><strong>Interactive Risk Exploration:</strong> Click-through disaster details</li>
                <li><strong>Real-time Updates:</strong> Dynamic data loading and visualization</li>
                <li><strong>Responsive Design:</strong> Cross-platform accessibility</li>
              </ul>
            </div>
          </div>
          
          <div style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: 'rgba(211, 47, 47, 0.05)', borderRadius: '8px', border: '1px solid rgba(211, 47, 47, 0.1)' }}>
            <h3 style={{ color: '#d32f2f', marginBottom: '1rem' }}>Pipeline Execution</h3>
            <p style={{ color: '#555', marginBottom: '1rem' }}>
              The system supports flexible execution modes with automated CSV generation for testing and development:
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
              <div>
                <strong style={{ color: '#333' }}>Required CSV Schema:</strong>
                <ul style={{ color: '#555', paddingLeft: '1.2rem', fontSize: '0.9rem', marginTop: '0.5rem' }}>
                  <li>Latitude, Longitude coordinates</li>
                  <li>Start/End dates (Year, Month, Day)</li>
                  <li>Disaster Type and Subtype classifications</li>
                </ul>
              </div>
              <div>
                <strong style={{ color: '#333' }}>Processing Steps:</strong>
                <ul style={{ color: '#555', paddingLeft: '1.2rem', fontSize: '0.9rem', marginTop: '0.5rem' }}>
                  <li>Automated data fetching from satellite catalogs</li>
                  <li>Spatial and temporal alignment of multi-source data</li>
                  <li>3D UNet training with multi-task objectives</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default MethodsPage;