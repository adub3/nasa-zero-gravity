import React, { useEffect, useState } from 'react';
import NavigationBar from './navigation-bar';
import { EarthquakesGeojson, loadEarthquakeGeojson } from './earthquakes';
import { loadGoogleMaps, getWildfireRiskColor, formatDate } from './map-utils';

// 3D Globe component for wildfires
const WildfireGlobe3D: React.FC<{ 
  wildfireData?: EarthquakesGeojson;
  onWildfireClick?: (wildfire: any) => void;
}> = ({ wildfireData, onWildfireClick }) => {
  const mapRef = React.useRef<HTMLDivElement>(null);
  const map3dRef = React.useRef<any>(null);
  const markersRef = React.useRef<any[]>([]);

  useEffect(() => {
    if (!mapRef.current) return;

    const initializeMap = async () => {
      try {
        await loadGoogleMaps();
        const { Map3DElement, MapMode, Marker3DElement } = await (window.google.maps as any).importLibrary("maps3d");
        const { PinElement } = await (window.google.maps as any).importLibrary("marker");

        const map3dElement = new Map3DElement({
          center: { lat: 39.8283, lng: -98.5795, altitude: 0 }, // Center on US for wildfire focus
          range: 8000000,
          tilt: 60,
          heading: 0,
          mode: MapMode.HYBRID
        });

        map3dRef.current = map3dElement;
        mapRef.current!.appendChild(map3dElement);

        window.Map3DElement = Map3DElement;
        window.Marker3DElement = Marker3DElement;
        window.PinElement = PinElement;

      } catch (error) {
        console.error('Failed to initialize 3D map:', error);
      }
    };

    initializeMap();

    return () => {
      markersRef.current.forEach(marker => {
        try {
          if (map3dRef.current && marker) {
            map3dRef.current.removeChild(marker);
          }
        } catch (e) {
          console.warn('Error removing marker');
        }
      });
      markersRef.current = [];

      if (mapRef.current && map3dRef.current) {
        try {
          mapRef.current.removeChild(map3dRef.current);
        } catch (e) {
          console.warn('Error removing 3D map element');
        }
      }
    };
  }, []);

  useEffect(() => {
    if (!map3dRef.current || !wildfireData || !window.Marker3DElement || !window.PinElement) return;

    const addMarkers = async () => {
      try {
        markersRef.current.forEach(marker => {
          try {
            if (marker && map3dRef.current) {
              map3dRef.current.removeChild(marker);
            }
          } catch (e) {
            console.warn('Error removing existing marker');
          }
        });
        markersRef.current = [];

        console.log(`Adding ${wildfireData.features.length} wildfire risk markers to globe`);
        
        wildfireData.features.slice(0, 200).forEach((wildfire, index) => {
          const [lng, lat] = wildfire.geometry.coordinates;
          const riskLevel = wildfire.properties?.mag || 1; // Using magnitude as risk placeholder
          
          try {
            const marker3d = new window.Marker3DElement({
              position: { 
                lat, 
                lng, 
                altitude: Math.max(1000, riskLevel * 30000)
              },
              altitudeMode: 'RELATIVE_TO_GROUND'
            });

            const scale = Math.max(0.4, Math.min(1.8, riskLevel / 4));
            const color = getWildfireRiskColor(riskLevel);
            
            const pinElement = new window.PinElement({
              background: color,
              borderColor: '#ffffff',
              glyphColor: '#ffffff',
              scale: scale
            });

            marker3d.appendChild(pinElement);
            marker3d.title = `Wildfire Risk Level: ${riskLevel} | Location: ${lat.toFixed(2)}, ${lng.toFixed(2)}`;
            
            marker3d.addEventListener('gmp-click', (event: any) => {
              if (onWildfireClick) {
                onWildfireClick(wildfire);
              }
              event.stopPropagation();
            });
            
            map3dRef.current.appendChild(marker3d);
            markersRef.current.push(marker3d);

          } catch (markerError) {
            console.warn(`Failed to create marker for wildfire ${index}:`, markerError);
          }
        });

        console.log(`Successfully added ${markersRef.current.length} wildfire markers to the globe`);
        
      } catch (error) {
        console.error('Error adding wildfire markers:', error);
      }
    };

    const timeout = setTimeout(() => {
      addMarkers();
    }, 1500);

    return () => clearTimeout(timeout);
  }, [wildfireData, onWildfireClick]);

  return (
    <div
      ref={mapRef}
      style={{
        width: '100%',
        height: '100%',
        position: 'relative'
      }}
    />
  );
};

// 2D Fallback for wildfires
const Wildfire2DFallback: React.FC<{ 
  wildfireData?: EarthquakesGeojson;
  onWildfireClick?: (wildfire: any) => void;
}> = ({ wildfireData, onWildfireClick }) => {
  const mapRef = React.useRef<HTMLDivElement>(null);
  const mapInstanceRef = React.useRef<google.maps.Map | null>(null);
  const markersRef = React.useRef<google.maps.Marker[]>([]);

  useEffect(() => {
    if (!mapRef.current || !window.google) return;

    const map = new google.maps.Map(mapRef.current, {
      center: { lat: 39.8283, lng: -98.5795 },
      zoom: 4,
      mapTypeId: 'satellite',
      tilt: 45,
      mapTypeControl: false,
      streetViewControl: false,
      fullscreenControl: false,
    });

    mapInstanceRef.current = map;

    return () => {
      markersRef.current.forEach(marker => marker.setMap(null));
      markersRef.current = [];
    };
  }, []);

  useEffect(() => {
    if (!mapInstanceRef.current || !wildfireData) return;

    markersRef.current.forEach(marker => marker.setMap(null));
    markersRef.current = [];

    wildfireData.features.slice(0, 300).forEach((wildfire) => {
      const [lng, lat] = wildfire.geometry.coordinates;
      const riskLevel = wildfire.properties?.mag || 1;
      
      const marker = new google.maps.Marker({
        position: { lat, lng },
        map: mapInstanceRef.current,
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: Math.max(4, riskLevel * 2.5),
          fillColor: getWildfireRiskColor(riskLevel),
          fillOpacity: 0.8,
          strokeWeight: 2,
          strokeColor: '#000000',
        },
        title: `Wildfire Risk Level: ${riskLevel}`
      });

      marker.addListener('click', () => {
        if (onWildfireClick) {
          onWildfireClick(wildfire);
        }
      });

      markersRef.current.push(marker);
    });
  }, [wildfireData, onWildfireClick]);

  return (
    <div
      ref={mapRef}
      style={{
        width: '100%',
        height: '100%'
      }}
    />
  );
};

const WildfireKey = () => {
  return (
    <div style={{
      position: 'fixed',
      bottom: '2rem',
      right: '2rem',
      background: 'rgba(255, 255, 255, 0.95)',
      backdropFilter: 'blur(10px)',
      borderRadius: '12px',
      padding: '1.5rem',
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
      border: '1px solid rgba(255, 255, 255, 0.2)',
      minWidth: '200px',
      zIndex: 999
    }}>
      <h4 style={{
        margin: '0 0 1rem 0',
        fontSize: '1rem',
        fontWeight: '600',
        color: '#333'
      }}>
        Wildfire Risk Level
      </h4>
      
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem'
      }}>
        {[
          { range: '7.0+', color: '#8B0000', label: 'Extreme' },
          { range: '6.0-6.9', color: '#FF0000', label: 'Very High' },
          { range: '5.0-5.9', color: '#FF4500', label: 'High' },
          { range: '4.0-4.9', color: '#FF8C00', label: 'Moderate' },
          { range: '3.0-3.9', color: '#FFD700', label: 'Low' },
          { range: '<3.0', color: '#FFFF00', label: 'Very Low' },
        ].map((item) => (
          <div key={item.range} style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontSize: '0.875rem'
          }}>
            <div style={{
              width: '12px',
              height: '12px',
              backgroundColor: item.color,
              borderRadius: '50%',
              border: '1px solid rgba(0, 0, 0, 0.2)'
            }} />
            <span style={{ color: '#666', minWidth: '50px' }}>{item.range}</span>
            <span style={{ color: '#888' }}>{item.label}</span>
          </div>
        ))}
      </div>
      
      <div style={{
        marginTop: '1rem',
        fontSize: '0.75rem',
        color: '#888',
        lineHeight: '1.4'
      }}>
        Wildfire risk assessment based on weather patterns, vegetation, and historical data
      </div>
    </div>
  );
};

const WildfiresPage = () => {
  const [wildfireData, setWildfireData] = useState<EarthquakesGeojson>();
  const [use3D, setUse3D] = useState(true);
  const [selectedWildfire, setSelectedWildfire] = useState<any>(null);

  useEffect(() => {
    // Using earthquake data as placeholder for wildfire data
    loadEarthquakeGeojson().then(data => setWildfireData(data));
  }, []);

  return (
    <div style={{ 
      width: '100vw', 
      height: '100vh',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Navigation */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '60px',
        background: 'rgba(255, 255, 255, 0.9)',
        backdropFilter: 'blur(10px)',
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 2rem',
        borderBottom: '1px solid rgba(0, 0, 0, 0.1)'
      }}>
        <NavigationBar />
      </div>
      
      {/* Toggle button */}
      <div style={{
        position: 'absolute',
        top: '80px',
        left: '2rem',
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(10px)',
        borderRadius: '12px',
        padding: '1rem',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
        border: '1px solid rgba(255, 255, 255, 0.2)',
        zIndex: 999
      }}>
        <button
          onClick={() => setUse3D(!use3D)}
          style={{
            background: use3D ? '#d32f2f' : '#666',
            color: 'white',
            border: 'none',
            padding: '0.5rem 1rem',
            borderRadius: '8px',
            cursor: 'pointer',
            fontSize: '0.9rem',
            fontWeight: '500'
          }}
        >
          {use3D ? '3D Globe' : '2D Satellite'}
        </button>
      </div>
      
      {/* Wildfire Information Popup */}
      {selectedWildfire && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          background: 'rgba(255, 255, 255, 0.98)',
          backdropFilter: 'blur(15px)',
          borderRadius: '16px',
          padding: '2rem',
          boxShadow: '0 16px 48px rgba(0, 0, 0, 0.2)',
          border: '1px solid rgba(255, 255, 255, 0.3)',
          zIndex: 1001,
          minWidth: '350px',
          maxWidth: '500px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
            <h3 style={{ 
              color: '#d32f2f', 
              margin: 0,
              fontSize: '1.5rem',
              fontWeight: '600'
            }}>
              Wildfire Risk Details
            </h3>
            <button
              onClick={() => setSelectedWildfire(null)}
              style={{
                background: 'none',
                border: 'none',
                fontSize: '1.5rem',
                color: '#666',
                cursor: 'pointer',
                padding: '0.25rem'
              }}
            >
              ×
            </button>
          </div>
          
          <div style={{ display: 'grid', gap: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: '600', color: '#333' }}>Risk Level:</span>
              <span style={{ 
                color: getWildfireRiskColor(selectedWildfire.properties.mag),
                fontWeight: '700',
                fontSize: '1.1rem'
              }}>
                {selectedWildfire.properties.mag || 'Unknown'}
              </span>
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: '600', color: '#333' }}>Location:</span>
              <span style={{ color: '#555', textAlign: 'right' }}>
                {selectedWildfire.geometry.coordinates[1].toFixed(3)}°N, {selectedWildfire.geometry.coordinates[0].toFixed(3)}°E
              </span>
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: '600', color: '#333' }}>Assessment Date:</span>
              <span style={{ color: '#555', textAlign: 'right' }}>
                {formatDate(selectedWildfire.properties.time)}
              </span>
            </div>
            
            <div style={{
              marginTop: '0.5rem',
              padding: '1rem',
              background: 'rgba(255, 140, 0, 0.1)',
              borderRadius: '8px',
              fontSize: '0.9rem',
              color: '#666',
              border: '1px solid rgba(255, 140, 0, 0.2)'
            }}>
              <strong>Note:</strong> This is placeholder data using earthquake information. Actual wildfire risk data will include vegetation moisture, weather patterns, and fire history.
            </div>
          </div>
        </div>
      )}
      
      {/* Globe container */}
      <div style={{ 
        width: '100%', 
        height: '100%',
        position: 'absolute',
        top: 0,
        left: 0
      }}>
        {use3D ? (
          <WildfireGlobe3D 
            wildfireData={wildfireData} 
            onWildfireClick={setSelectedWildfire}
          />
        ) : (
          <Wildfire2DFallback 
            wildfireData={wildfireData}
            onWildfireClick={setSelectedWildfire}
          />
        )}
      </div>
      
      {/* Legend */}
      <div style={{
        position: 'absolute',
        bottom: '2rem',
        right: '2rem',
        zIndex: 999
      }}>
        <WildfireKey />
      </div>
    </div>
  );
};

export default WildfiresPage;

// 3D Globe component for wildfires
const WildfireGlobe3D: React.FC<{ 
  wildfireData?: EarthquakesGeojson;
  onWildfireClick?: (wildfire: any) => void;
}> = ({ wildfireData, onWildfireClick }) => {
  const mapRef = React.useRef<HTMLDivElement>(null);
  const map3dRef = React.useRef<any>(null);
  const markersRef = React.useRef<any[]>([]);

  useEffect(() => {
    if (!mapRef.current) return;

    const initializeMap = async () => {
      try {
        await loadGoogleMaps();
        const { Map3DElement, MapMode, Marker3DElement } = await (window.google.maps as any).importLibrary("maps3d");
        const { PinElement } = await (window.google.maps as any).importLibrary("marker");

        const map3dElement = new Map3DElement({
          center: { lat: 39.8283, lng: -98.5795, altitude: 0 }, // Center on US for wildfire focus
          range: 8000000,
          tilt: 60,
          heading: 0,
          mode: MapMode.HYBRID
        });

        map3dRef.current = map3dElement;
        mapRef.current!.appendChild(map3dElement);

        window.Map3DElement = Map3DElement;
        window.Marker3DElement = Marker3DElement;
        window.PinElement = PinElement;

      } catch (error) {
        console.error('Failed to initialize 3D map:', error);
      }
    };

    initializeMap();

    return () => {
      markersRef.current.forEach(marker => {
        try {
          if (map3dRef.current && marker) {
            map3dRef.current.removeChild(marker);
          }
        } catch (e) {
          console.warn('Error removing marker');
        }
      });
      markersRef.current = [];

      if (mapRef.current && map3dRef.current) {
        try {
          mapRef.current.removeChild(map3dRef.current);
        } catch (e) {
          console.warn('Error removing 3D map element');
        }
      }
    };
  }, []);

  useEffect(() => {
    if (!map3dRef.current || !wildfireData || !window.Marker3DElement || !window.PinElement) return;

    const addMarkers = async () => {
      try {
        markersRef.current.forEach(marker => {
          try {
            if (marker && map3dRef.current) {
              map3dRef.current.removeChild(marker);
            }
          } catch (e) {
            console.warn('Error removing existing marker');
          }
        });
        markersRef.current = [];

        console.log(`Adding ${wildfireData.features.length} wildfire risk markers to globe`);
        
        wildfireData.features.slice(0, 200).forEach((wildfire, index) => {
          const [lng, lat] = wildfire.geometry.coordinates;
          const riskLevel = wildfire.properties?.mag || 1; // Using magnitude as risk placeholder
          
          try {
            const marker3d = new window.Marker3DElement({
              position: { 
                lat, 
                lng, 
                altitude: Math.max(1000, riskLevel * 30000)
              },
              altitudeMode: 'RELATIVE_TO_GROUND'
            });

            const scale = Math.max(0.4, Math.min(1.8, riskLevel / 4));
            const color = getWildfireRiskColor(riskLevel);
            
            const pinElement = new window.PinElement({
              background: color,
              borderColor: '#ffffff',
              glyphColor: '#ffffff',
              scale: scale
            });

            marker3d.appendChild(pinElement);
            marker3d.title = `Wildfire Risk Level: ${riskLevel} | Location: ${lat.toFixed(2)}, ${lng.toFixed(2)}`;
            
            marker3d.addEventListener('gmp-click', (event: any) => {
              if (onWildfireClick) {
                onWildfireClick(wildfire);
              }
              event.stopPropagation();
            });
            
            map3dRef.current.appendChild(marker3d);
            markersRef.current.push(marker3d);

          } catch (markerError) {
            console.warn(`Failed to create marker for wildfire ${index}:`, markerError);
          }
        });

        console.log(`Successfully added ${markersRef.current.length} wildfire markers to the globe`);
        
      } catch (error) {
        console.error('Error adding wildfire markers:', error);
      }
    };

    const timeout = setTimeout(() => {
      addMarkers();
    }, 1500);

    return () => clearTimeout(timeout);
  }, [wildfireData, onWildfireClick]);

  return (
    <div
      ref={mapRef}
      style={{
        width: '100%',
        height: '100%',
        position: 'relative'
      }}
    />
  );
};

// 2D Fallback for wildfires
const Wildfire2DFallback: React.FC<{ 
  wildfireData?: EarthquakesGeojson;
  onWildfireClick?: (wildfire: any) => void;
}> = ({ wildfireData, onWildfireClick }) => {
  const mapRef = React.useRef<HTMLDivElement>(null);
  const mapInstanceRef = React.useRef<google.maps.Map | null>(null);
  const markersRef = React.useRef<google.maps.Marker[]>([]);

  useEffect(() => {
    if (!mapRef.current || !window.google) return;

    const map = new google.maps.Map(mapRef.current, {
      center: { lat: 39.8283, lng: -98.5795 },
      zoom: 4,
      mapTypeId: 'satellite',
      tilt: 45,
      mapTypeControl: false,
      streetViewControl: false,
      fullscreenControl: false,
    });

    mapInstanceRef.current = map;

    return () => {
      markersRef.current.forEach(marker => marker.setMap(null));
      markersRef.current = [];
    };
  }, []);

  useEffect(() => {
    if (!mapInstanceRef.current || !wildfireData) return;

    markersRef.current.forEach(marker => marker.setMap(null));
    markersRef.current = [];

    wildfireData.features.slice(0, 300).forEach((wildfire) => {
      const [lng, lat] = wildfire.geometry.coordinates;
      const riskLevel = wildfire.properties?.mag || 1;
      
      const marker = new google.maps.Marker({
        position: { lat, lng },
        map: mapInstanceRef.current,
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: Math.max(4, riskLevel * 2.5),
          fillColor: getWildfireRiskColor(riskLevel),
          fillOpacity: 0.8,
          strokeWeight: 2,
          strokeColor: '#000000',
        },
        title: `Wildfire Risk Level: ${riskLevel}`
      });

      marker.addListener('click', () => {
        if (onWildfireClick) {
          onWildfireClick(wildfire);
        }
      });

      markersRef.current.push(marker);
    });
  }, [wildfireData, onWildfireClick]);

  return (
    <div
      ref={mapRef}
      style={{
        width: '100%',
        height: '100%'
      }}
    />
  );
};

const WildfireKey = () => {
  return (
    <div style={{
      position: 'fixed',
      bottom: '2rem',
      right: '2rem',
      background: 'rgba(255, 255, 255, 0.95)',
      backdropFilter: 'blur(10px)',
      borderRadius: '12px',
      padding: '1.5rem',
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
      border: '1px solid rgba(255, 255, 255, 0.2)',
      minWidth: '200px',
      zIndex: 999
    }}>
      <h4 style={{
        margin: '0 0 1rem 0',
        fontSize: '1rem',
        fontWeight: '600',
        color: '#333'
      }}>
        Wildfire Risk Level
      </h4>
      
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem'
      }}>
        {[
          { range: '7.0+', color: '#8B0000', label: 'Extreme' },
          { range: '6.0-6.9', color: '#FF0000', label: 'Very High' },
          { range: '5.0-5.9', color: '#FF4500', label: 'High' },
          { range: '4.0-4.9', color: '#FF8C00', label: 'Moderate' },
          { range: '3.0-3.9', color: '#FFD700', label: 'Low' },
          { range: '<3.0', color: '#FFFF00', label: 'Very Low' },
        ].map((item) => (
          <div key={item.range} style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontSize: '0.875rem'
          }}>
            <div style={{
              width: '12px',
              height: '12px',
              backgroundColor: item.color,
              borderRadius: '50%',
              border: '1px solid rgba(0, 0, 0, 0.2)'
            }} />
            <span style={{ color: '#666', minWidth: '50px' }}>{item.range}</span>
            <span style={{ color: '#888' }}>{item.label}</span>
          </div>
        ))}
      </div>
      
      <div style={{
        marginTop: '1rem',
        fontSize: '0.75rem',
        color: '#888',
        lineHeight: '1.4'
      }}>
        Wildfire risk assessment based on weather patterns, vegetation, and historical data
      </div>
    </div>
  );
};

const WildfiresPage = () => {
  const [wildfireData, setWildfireData] = useState<EarthquakesGeojson>();
  const [use3D, setUse3D] = useState(true);
  const [selectedWildfire, setSelectedWildfire] = useState<any>(null);

  useEffect(() => {
    // Using earthquake data as placeholder for wildfire data
    loadEarthquakeGeojson().then(data => setWildfireData(data));
  }, []);

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div style={{ 
      width: '100vw', 
      height: '100vh',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Navigation */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '60px',
        background: 'rgba(255, 255, 255, 0.9)',
        backdropFilter: 'blur(10px)',
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 2rem',
        borderBottom: '1px solid rgba(0, 0, 0, 0.1)'
      }}>
        <NavigationBar />
      </div>
      
      {/* Toggle button */}
      <div style={{
        position: 'absolute',
        top: '80px',
        left: '2rem',
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(10px)',
        borderRadius: '12px',
        padding: '1rem',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
        border: '1px solid rgba(255, 255, 255, 0.2)',
        zIndex: 999
      }}>
        <button
          onClick={() => setUse3D(!use3D)}
          style={{
            background: use3D ? '#d32f2f' : '#666',
            color: 'white',
            border: 'none',
            padding: '0.5rem 1rem',
            borderRadius: '8px',
            cursor: 'pointer',
            fontSize: '0.9rem',
            fontWeight: '500'
          }}
        >
          {use3D ? '3D Globe' : '2D Satellite'}
        </button>
      </div>
      
      {/* Wildfire Information Popup */}
      {selectedWildfire && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          background: 'rgba(255, 255, 255, 0.98)',
          backdropFilter: 'blur(15px)',
          borderRadius: '16px',
          padding: '2rem',
          boxShadow: '0 16px 48px rgba(0, 0, 0, 0.2)',
          border: '1px solid rgba(255, 255, 255, 0.3)',
          zIndex: 1001,
          minWidth: '350px',
          maxWidth: '500px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
            <h3 style={{ 
              color: '#d32f2f', 
              margin: 0,
              fontSize: '1.5rem',
              fontWeight: '600'
            }}>
              Wildfire Risk Details
            </h3>
            <button
              onClick={() => setSelectedWildfire(null)}
              style={{
                background: 'none',
                border: 'none',
                fontSize: '1.5rem',
                color: '#666',
                cursor: 'pointer',
                padding: '0.25rem'
              }}
            >
              ×
            </button>
          </div>
          
          <div style={{ display: 'grid', gap: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: '600', color: '#333' }}>Risk Level:</span>
              <span style={{ 
                color: getWildfireRiskColor(selectedWildfire.properties.mag),
                fontWeight: '700',
                fontSize: '1.1rem'
              }}>
                {selectedWildfire.properties.mag || 'Unknown'}
              </span>
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: '600', color: '#333' }}>Location:</span>
              <span style={{ color: '#555', textAlign: 'right' }}>
                {selectedWildfire.geometry.coordinates[1].toFixed(3)}°N, {selectedWildfire.geometry.coordinates[0].toFixed(3)}°E
              </span>
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: '600', color: '#333' }}>Assessment Date:</span>
              <span style={{ color: '#555', textAlign: 'right' }}>
                {formatDate(selectedWildfire.properties.time)}
              </span>
            </div>
            
            <div style={{
              marginTop: '0.5rem',
              padding: '1rem',
              background: 'rgba(255, 140, 0, 0.1)',
              borderRadius: '8px',
              fontSize: '0.9rem',
              color: '#666',
              border: '1px solid rgba(255, 140, 0, 0.2)'
            }}>
              <strong>Note:</strong> This is placeholder data using earthquake information. Actual wildfire risk data will include vegetation moisture, weather patterns, and fire history.
            </div>
          </div>
        </div>
      )}
      
      {/* Globe container */}
      <div style={{ 
        width: '100%', 
        height: '100%',
        position: 'absolute',
        top: 0,
        left: 0
      }}>
        {use3D ? (
          <WildfireGlobe3D 
            wildfireData={wildfireData} 
            onWildfireClick={setSelectedWildfire}
          />
        ) : (
          <Wildfire2DFallback 
            wildfireData={wildfireData}
            onWildfireClick={setSelectedWildfire}
          />
        )}
      </div>
      
      {/* Legend */}
      <div style={{
        position: 'absolute',
        bottom: '2rem',
        right: '2rem',
        zIndex: 999
      }}>
        <WildfireKey />
      </div>
    </div>
  );
};

export default WildfiresPage;