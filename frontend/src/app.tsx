import React, {useEffect, useState} from 'react';
import {BrowserRouter as Router, Routes, Route} from 'react-router-dom';

import {APIProvider, Map} from '@vis.gl/react-google-maps';

import NavigationBar from './navigation-bar';
import HeatmapKey from './heatmap-key';
import Heatmap from './heatmap';
import MethodsPage from './methods';
import AboutPage from './about';
import {EarthquakesGeojson, loadEarthquakeGeojson} from './earthquakes';

const API_KEY =
  globalThis.GOOGLE_MAPS_API_KEY ?? (process.env.GOOGLE_MAPS_API_KEY as string);

// Main map component
const MapPage = () => {
  const [earthquakesGeojson, setEarthquakesGeojson] =
    useState<EarthquakesGeojson>();

  useEffect(() => {
    loadEarthquakeGeojson().then(data => setEarthquakesGeojson(data));
  }, []);

  return (
    <>
      <NavigationBar />
      
      <APIProvider apiKey={API_KEY}>
        <Map
          mapId={'7a9e2ebecd32a903'}
          defaultCenter={{lat: 40.7749, lng: -130.4194}}
          defaultZoom={3}
          gestureHandling={'greedy'}
          disableDefaultUI={true}
          mapTypeId="satellite"
        />

        {earthquakesGeojson && (
          <Heatmap
            geojson={earthquakesGeojson}
            radius={25}
            opacity={0.8}
          />
        )}

        <HeatmapKey />
      </APIProvider>
    </>
  );
};

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MapPage />} />
        <Route path="/methods" element={<MethodsPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </Router>
  );
};

export default App;

// Add this to the end of your app.tsx
export const renderToDom = (container: HTMLElement) => {
  const root = ReactDOM.createRoot(container);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
};