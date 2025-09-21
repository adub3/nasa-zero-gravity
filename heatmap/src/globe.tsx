import React, { useEffect, useState } from 'react';
import NavigationBar from './navigation-bar';
import { DisasterRiskGeojson, loadWildfireData, loadFloodData, loadHurricaneData } from './disaster-data';

const API_KEY = globalThis.GOOGLE_MAPS_API_KEY ?? (process.env.GOOGLE_MAPS_API_KEY as string);

// Load Google Maps API with 3D support
const loadGoogleMaps = () => {
  return new Promise<void>((resolve, reject) => {
    if (window.google && window.google.maps) {
      resolve();
      return;
    }

    const script = document.createElement('script');
    script.src = `https://maps.googleapis.com/maps/api/js?key=${API_KEY}&v=alpha&libraries=maps3d`;
    script.async = true;
    script.defer = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('Google Maps API could not be loaded'));
    document.head.appendChild(script);
  });
};

// Helper function to get color based on probability (0-1)
const getProbabilityColor = (probability: number, disasterType: string): string => {
  // Clamp probability to 0-1 range
  const p = Math.max(0, Math.min(1, probability));
  
  if (disasterType === 'wildfire') {
    // Red gradient for wildfires
    if (p >= 0.8) return '#8B0000'; // Dark red
    if (p >= 0.6) return '#FF0000'; // Red
    if (p >= 0.4) return '#FF4500'; // Orange-red
    if (p >= 0.2) return '#FF8C00'; // Orange
    return '#FFD700'; // Gold
  } else if (disasterType === 'flood') {
    // Blue gradient for floods
    if (p >= 0.8) return '#000080'; // Dark blue
    if (p >= 0.6) return '#0000FF'; // Blue
    if (p >= 0.4) return '#4169E1'; // Royal blue
    if (p >= 0.2) return '#87CEEB'; // Sky blue
    return '#ADD8E6'; // Light blue
  } else if (disasterType === 'hurricane') {
    // Purple/pink gradient for hurricanes
    if (p >= 0.8) return '#8B008B'; // Dark magenta
    if (p >= 0.6) return '#FF1493'; // Deep pink
    if (p >= 0.4) return '#FF69B4'; // Hot pink
    if (p >= 0.2) return '#FFA500'; // Orange
    return '#FFD700'; // Gold
  }
  
  return '#808080'; // Gray fallback
};

// Custom 3D Map component
const Globe3D: React.FC<{ 
  disasterData?: DisasterRiskGeojson;
  onDisasterClick?: (disaster: any) => void;
  isActive: boolean;
  disasterType: string;
}> = ({ disasterData, onDisasterClick, isActive, disasterType }) => {
  const mapRef = React.useRef<HTMLDivElement>(null);
  const map3dRef = React.useRef<any>(null);
  const markersRef = React.useRef<any[]>([]);

  useEffect(() => {
    if (!mapRef.current || !isActive) return;

    const initializeMap = async () => {
      try {
        await loadGoogleMaps();
        const { Map3DElement, MapMode, Marker3DElement } = await (window.google.maps as any).importLibrary("maps3d");
        const { PinElement } = await (window.google.maps as any).importLibrary("marker");

        const map3dElement = new Map3DElement({
          center: { lat: 35.6762, lng: 139.6503, altitude: 0 },
          range: 10000000,
          tilt: 60,
          heading: 0,
          mode: MapMode.HYBRID
        });

        map3dRef.current = map3dElement;
        mapRef.current!.appendChild(map3dElement);

        (window as any).Map3DElement = Map3DElement;
        (window as any).Marker3DElement = Marker3DElement;
        (window as any).PinElement = PinElement;

        map3dElement.addEventListener('gmp-click', (event: any) => {
          console.log("Globe clicked at:", event.position);
        });

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
  }, [isActive]);

  useEffect(() => {
    if (!map3dRef.current || !disasterData || !(window as any).Marker3DElement || !(window as any).PinElement || !isActive) return;

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

        console.log(`Adding ${disasterData.features.length} ${disasterType} risk markers to 3D globe`);
        
        disasterData.features.slice(0, 200).forEach((disaster, index) => {
          const [lng, lat] = disaster.geometry.coordinates;
          const probability = disaster.properties?.probability || 0;
          
          try {
            const marker3d = new (window as any).Marker3DElement({
              position: { 
                lat, 
                lng, 
                altitude: Math.max(1000, probability * 50000)
              },
              altitudeMode: 'RELATIVE_TO_GROUND'
            });

            const scale = Math.max(0.3, Math.min(1.5, probability * 2));
            const color = getProbabilityColor(probability, disasterType);
            
            const pinElement = new (window as any).PinElement({
              background: color,
              borderColor: '#ffffff',
              glyphColor: '#ffffff',
              scale: scale
            });

            marker3d.appendChild(pinElement);
            marker3d.title = `${disasterType} Risk: ${(probability * 100).toFixed(1)}% | Location: ${lat.toFixed(2)}, ${lng.toFixed(2)}`;
            
            marker3d.addEventListener('gmp-click', (event: any) => {
              if (onDisasterClick) {
                onDisasterClick(disaster);
              }
              event.stopPropagation();
            });
            
            map3dRef.current.appendChild(marker3d);
            markersRef.current.push(marker3d);

          } catch (markerError) {
            console.warn(`Failed to create marker for ${disasterType} ${index}:`, markerError);
          }
        });

        console.log(`Successfully added ${markersRef.current.length} markers to the 3D globe`);
        
      } catch (error) {
        console.error(`Error adding ${disasterType} markers:`, error);
      }
    };

    const timeout = setTimeout(() => {
      addMarkers();
    }, 1500);

    return () => clearTimeout(timeout);
  }, [disasterData, onDisasterClick, isActive, disasterType]);

  return (
    <div
      ref={mapRef}
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        display: isActive ? 'block' : 'none'
      }}
    />
  );
};

// 2D Satellite Map component
const Satellite2D: React.FC<{ 
  disasterData?: DisasterRiskGeojson;
  onDisasterClick?: (disaster: any) => void;
  isActive: boolean;
  disasterType: string;
}> = ({ disasterData, onDisasterClick, isActive, disasterType }) => {
  const mapRef = React.useRef<HTMLDivElement>(null);
  const mapInstanceRef = React.useRef<google.maps.Map | null>(null);
  const markersRef = React.useRef<google.maps.Marker[]>([]);

  useEffect(() => {
    if (!mapRef.current || !window.google || !isActive) return;

    const map = new google.maps.Map(mapRef.current, {
      center: { lat: 20, lng: 0 },
      zoom: 2,
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
  }, [isActive]);

  useEffect(() => {
    if (!mapInstanceRef.current || !disasterData || !isActive) return;

    markersRef.current.forEach(marker => marker.setMap(null));
    markersRef.current = [];

    console.log(`Adding ${disasterData.features.length} ${disasterType} risk markers to 2D satellite map`);

    disasterData.features.slice(0, 300).forEach((disaster) => {
      const [lng, lat] = disaster.geometry.coordinates;
      const probability = disaster.properties?.probability || 0;
      
      const marker = new google.maps.Marker({
        position: { lat, lng },
        map: mapInstanceRef.current,
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: Math.max(3, probability * 15),
          fillColor: getProbabilityColor(probability, disasterType),
          fillOpacity: 0.8,
          strokeWeight: 1,
          strokeColor: '#000000',
        },
        title: `${disasterType} Risk: ${(probability * 100).toFixed(1)}%`
      });

      marker.addListener('click', () => {
        if (onDisasterClick) {
          onDisasterClick(disaster);
        }
      });

      markersRef.current.push(marker);
    });

    console.log(`Successfully added ${markersRef.current.length} markers to the 2D satellite map`);
  }, [disasterData, onDisasterClick, isActive, disasterType]);

  return (
    <div
      ref={mapRef}
      style={{
        width: '100%',
        height: '100%',
        display: isActive ? 'block' : 'none'
      }}
    />
  );
};

const DisasterKey: React.FC<{ disasterType: string }> = ({ disasterType }) => {
  const getColorScheme = () => {
    if (disasterType === 'wildfire') {
      return [
        { range: '80-100%', color: '#8B0000', label: 'Very High' },
        { range: '60-80%', color: '#FF0000', label: 'High' },
        { range: '40-60%', color: '#FF4500', label: 'Moderate' },
        { range: '20-40%', color: '#FF8C00', label: 'Low' },
        { range: '0-20%', color: '#FFD700', label: 'Very Low' },
      ];
    } else if (disasterType === 'flood') {
      return [
        { range: '80-100%', color: '#000080', label: 'Very High' },
        { range: '60-80%', color: '#0000FF', label: 'High' },
        { range: '40-60%', color: '#4169E1', label: 'Moderate' },
        { range: '20-40%', color: '#87CEEB', label: 'Low' },
        { range: '0-20%', color: '#ADD8E6', label: 'Very Low' },
      ];
    } else if (disasterType === 'hurricane') {
      return [
        { range: '80-100%', color: '#8B008B', label: 'Very High' },
        { range: '60-80%', color: '#FF1493', label: 'High' },
        { range: '40-60%', color: '#FF69B4', label: 'Moderate' },
        { range: '20-40%', color: '#FFA500', label: 'Low' },
        { range: '0-20%', color: '#FFD700', label: 'Very Low' },
      ];
    }
    return [];
  };

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
        color: '#333',
        textTransform: 'capitalize'
      }}>
        {disasterType} Risk Probability
      </h4>
      
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem'
      }}>
        {getColorScheme().map((item) => (
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
            <span style={{ color: '#666', minWidth: '60px' }}>{item.range}</span>
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
        Risk probability based on predictive models and environmental factors
      </div>
    </div>
  );
};

const GlobePage = () => {
  const [disasterData, setDisasterData] = useState<DisasterRiskGeojson>();
  const [use3D, setUse3D] = useState(true);
  const [selectedDisaster, setSelectedDisaster] = useState<any>(null);
  const [disasterType, setDisasterType] = useState<'wildfire' | 'flood' | 'hurricane'>('wildfire');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        let data;
        switch (disasterType) {
          case 'wildfire':
            data = await loadWildfireData();
            break;
          case 'flood':
            data = await loadFloodData();
            break;
          case 'hurricane':
            data = await loadHurricaneData();
            break;
        }
        setDisasterData(data);
      } catch (error) {
        console.error(`Error loading ${disasterType} data:`, error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [disasterType]);

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
      <NavigationBar />
      
      <div style={{
        position: 'absolute',
        top: '80px',
        left: '2rem',
        display: 'flex',
        gap: '1rem',
        zIndex: 999
      }}>
        <div style={{
          background: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(10px)',
          borderRadius: '12px',
          padding: '1rem',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          border: '1px solid rgba(255, 255, 255, 0.2)'
        }}>
          <select
            value={disasterType}
            onChange={(e) => setDisasterType(e.target.value as 'wildfire' | 'flood' | 'hurricane')}
            style={{
              background: 'transparent',
              border: 'none',
              fontSize: '0.9rem',
              fontWeight: '500',
              color: '#333',
              cursor: 'pointer',
              outline: 'none',
              textTransform: 'capitalize'
            }}
          >
            <option value="wildfire">Wildfires</option>
            <option value="flood">Floods</option>
            <option value="hurricane">Hurricanes</option>
          </select>
        </div>

        <div style={{
          background: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(10px)',
          borderRadius: '12px',
          padding: '1rem',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          border: '1px solid rgba(255, 255, 255, 0.2)'
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
      </div>
      
      {loading && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          background: 'rgba(255, 255, 255, 0.9)',
          padding: '1rem 2rem',
          borderRadius: '12px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          zIndex: 1002,
          display: 'flex',
          alignItems: 'center',
          gap: '1rem'
        }}>
          <div style={{
            width: '20px',
            height: '20px',
            border: '2px solid #d32f2f',
            borderTop: '2px solid transparent',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }} />
          <span>Loading {disasterType} data...</span>
        </div>
      )}
      
      {selectedDisaster && (
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
              color: getProbabilityColor(selectedDisaster.properties.probability, disasterType),
              margin: 0,
              fontSize: '1.5rem',
              fontWeight: '600',
              textTransform: 'capitalize'
            }}>
              {disasterType} Risk Details
            </h3>
            <button
              onClick={() => setSelectedDisaster(null)}
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
              <span style={{ fontWeight: '600', color: '#333' }}>Risk Probability:</span>
              <span style={{ 
                color: getProbabilityColor(selectedDisaster.properties.probability, disasterType),
                fontWeight: '700',
                fontSize: '1.1rem'
              }}>
                {(selectedDisaster.properties.probability * 100).toFixed(1)}%
              </span>
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: '600', color: '#333' }}>Location:</span>
              <span style={{ color: '#555', textAlign: 'right' }}>
                {selectedDisaster.geometry.coordinates[1].toFixed(4)}°N, {Math.abs(selectedDisaster.geometry.coordinates[0]).toFixed(4)}°{selectedDisaster.geometry.coordinates[0] >= 0 ? 'E' : 'W'}
              </span>
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: '600', color: '#333' }}>Last Assessment:</span>
              <span style={{ color: '#555', textAlign: 'right' }}>
                {formatDate(selectedDisaster.properties.lastAssessment)}
              </span>
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: '600', color: '#333' }}>Risk Level:</span>
              <span style={{ color: '#555' }}>
                {selectedDisaster.properties.probability >= 0.8 ? 'Very High' :
                 selectedDisaster.properties.probability >= 0.6 ? 'High' :
                 selectedDisaster.properties.probability >= 0.4 ? 'Moderate' :
                 selectedDisaster.properties.probability >= 0.2 ? 'Low' : 'Very Low'}
              </span>
            </div>
            
            <div style={{
              marginTop: '0.5rem',
              padding: '1rem',
              background: 'rgba(108, 108, 108, 0.1)',
              borderRadius: '8px',
              fontSize: '0.9rem',
              color: '#666',
              border: '1px solid rgba(108, 108, 108, 0.2)'
            }}>
              <strong>Model ID:</strong> {selectedDisaster.properties.id}
            </div>
          </div>
        </div>
      )}
      
      <div style={{ 
        width: '100%', 
        height: '100%',
        position: 'absolute',
        top: 0,
        left: 0
      }}>
        <Globe3D 
          disasterData={disasterData} 
          onDisasterClick={setSelectedDisaster}
          isActive={use3D}
          disasterType={disasterType}
        />
        <Satellite2D 
          disasterData={disasterData}
          onDisasterClick={setSelectedDisaster}
          isActive={!use3D}
          disasterType={disasterType}
        />
      </div>
      
      <DisasterKey disasterType={disasterType} />

      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
};

export default GlobePage;