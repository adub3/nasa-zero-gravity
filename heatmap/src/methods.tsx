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
            maxWidth: '600px',
            margin: '0 auto'
          }}>
            Understanding how we analyze and visualize earthquake risk data
          </p>
        </header>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
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
              <li>USGS Earthquake Hazards Program</li>
              <li>Global Earthquake Model (GEM)</li>
              <li>National Seismic Hazard Maps</li>
              <li>Historical seismic records dating back to 1900</li>
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
            <h2 style={{ color: '#d32f2f', marginBottom: '1rem' }}>Risk Calculation</h2>
            <p style={{ color: '#555', marginBottom: '1rem' }}>
              Risk levels are calculated using a weighted combination of:
            </p>
            <ul style={{ color: '#555', paddingLeft: '1.2rem' }}>
              <li>Earthquake magnitude (Richter scale)</li>
              <li>Frequency of occurrence</li>
              <li>Population density</li>
              <li>Infrastructure vulnerability</li>
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
          <h2 style={{ color: '#d32f2f', marginBottom: '1.5rem' }}>Visualization Methodology</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', alignItems: 'start' }}>
            <div>
              <h3 style={{ color: '#333', marginBottom: '1rem' }}>Heatmap Generation</h3>
              <p style={{ color: '#555', marginBottom: '1rem' }}>
                Our heatmap visualization uses the Google Maps JavaScript API's visualization library 
                to create density overlays based on processed earthquake data.
              </p>
              <h4 style={{ color: '#333', marginBottom: '0.5rem' }}>Key Parameters:</h4>
              <ul style={{ color: '#555', paddingLeft: '1.2rem' }}>
                <li><strong>Radius:</strong> 25km smoothing radius</li>
                <li><strong>Opacity:</strong> 0.8 for optimal visibility</li>
                <li><strong>Gradient:</strong> Transparent to red scale</li>
                <li><strong>Weight:</strong> Based on magnitude squared</li>
              </ul>
            </div>
            
            <div>
              <h3 style={{ color: '#333', marginBottom: '1rem' }}>Color Coding System</h3>
              <div style={{
                background: 'linear-gradient(to right, rgba(255, 0, 0, 0), rgba(255, 0, 0, 1))',
                height: '30px',
                borderRadius: '15px',
                border: '1px solid rgba(0, 0, 0, 0.1)',
                marginBottom: '1rem'
              }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                <span>Low Risk</span>
                <span>High Risk</span>
              </div>
              <p style={{ color: '#555' }}>
                The gradient represents normalized risk scores from 0 (transparent) 
                to 1 (solid red), allowing for immediate visual assessment of relative earthquake risk.
              </p>
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
          <h2 style={{ color: '#d32f2f', marginBottom: '1.5rem' }}>Technical Implementation</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '2rem' }}>
            <div>
              <h3 style={{ color: '#333', marginBottom: '1rem' }}>Frontend Technologies</h3>
              <ul style={{ color: '#555', paddingLeft: '1.2rem' }}>
                <li>React 19 with TypeScript</li>
                <li>@vis.gl/react-google-maps</li>
                <li>Google Maps JavaScript API</li>
                <li>Vite build system</li>
              </ul>
            </div>
            
            <div>
              <h3 style={{ color: '#333', marginBottom: '1rem' }}>Data Processing</h3>
              <ul style={{ color: '#555', paddingLeft: '1.2rem' }}>
                <li>GeoJSON format processing</li>
                <li>Real-time data aggregation</li>
                <li>Spatial indexing for performance</li>
                <li>Statistical normalization</li>
              </ul>
            </div>
            
            <div>
              <h3 style={{ color: '#333', marginBottom: '1rem' }}>Performance</h3>
              <ul style={{ color: '#555', paddingLeft: '1.2rem' }}>
                <li>Client-side rendering optimization</li>
                <li>Efficient memory management</li>
                <li>Lazy loading of map tiles</li>
                <li>Responsive design principles</li>
              </ul>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default MethodsPage;