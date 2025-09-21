import React, { useEffect, useState } from 'react';
import NavigationBar from './navigation-bar';
import { FeatureCollection, Point } from 'geojson';

export type DisasterRiskProps = {
  id: string;
  probability: number; // 0-1 probability value
  riskType: 'wildfire' | 'drought';
  lastAssessment: number; // timestamp
};

export type DisasterRiskGeojson = FeatureCollection<Point, DisasterRiskProps>;

const API_KEY = globalThis.GOOGLE_MAPS_API_KEY ?? (process.env.GOOGLE_MAPS_API_KEY as string);

// Generate grid-based risk data around specific hotspots
// Simulated data copied directly from Claude
const generateRiskGrid = (
  centerLat: number,
  centerLng: number,
  gridSize: number,
  spacing: number,
  riskType: 'wildfire' | 'drought',
  baseRisk: number = 0.5
): DisasterRiskProps[] => {
  const points: DisasterRiskProps[] = [];
  const halfGrid = Math.floor(gridSize / 2);
  
  for (let i = -halfGrid; i <= halfGrid; i++) {
    for (let j = -halfGrid; j <= halfGrid; j++) {
      const lat = centerLat + (i * spacing);
      const lng = centerLng + (j * spacing);
      
      // Distance from center affects probability
      const distance = Math.sqrt(i * i + j * j);
      const maxDistance = halfGrid * Math.sqrt(2);
      const distanceFactor = 1 - (distance / maxDistance);
      
      // Add some randomness while maintaining center-high pattern
      const randomFactor = 0.3 + (Math.random() * 0.4); // 0.3 to 0.7
      const probability = Math.max(0.05, Math.min(0.95, 
        baseRisk * distanceFactor * randomFactor + (Math.random() * 0.2 - 0.1)
      ));
      
      points.push({
        id: `${riskType}_${lat.toFixed(4)}_${lng.toFixed(4)}`,
        probability: parseFloat(probability.toFixed(3)),
        riskType,
        lastAssessment: Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000 // Random time in last 30 days
      });
    }
  }
  
  return points;
};

// Generate realistic wildfire risk data (focused on known fire-prone land regions only)
const generateWildfireData = (): DisasterRiskGeojson => {
  const wildfirePoints: DisasterRiskProps[] = [];
  
  // California inland areas (avoiding coastal waters)
  const californiaPoints = [
    { lat: 34.0522, lng: -118.2437, risk: 0.8 }, // LA inland
    { lat: 34.2, lng: -118.5, risk: 0.75 },
    { lat: 34.4, lng: -118.1, risk: 0.7 },
    { lat: 37.7749, lng: -122.4194, risk: 0.85 }, // Bay Area inland
    { lat: 37.9, lng: -122.2, risk: 0.8 },
    { lat: 37.6, lng: -122.1, risk: 0.75 },
    { lat: 38.5816, lng: -121.4944, risk: 0.9 }, // Sacramento Valley
    { lat: 38.8, lng: -121.2, risk: 0.85 },
    { lat: 39.0, lng: -121.0, risk: 0.8 },
    { lat: 36.7783, lng: -119.4179, risk: 0.7 }, // Central Valley inland
    { lat: 36.5, lng: -119.2, risk: 0.75 },
    { lat: 37.0, lng: -119.6, risk: 0.8 },
  ];

  // Pacific Northwest inland areas
  const pacificNWPoints = [
    { lat: 45.5152, lng: -122.6784, risk: 0.65 }, // Portland area
    { lat: 45.7, lng: -122.4, risk: 0.6 },
    { lat: 45.3, lng: -122.3, risk: 0.7 },
    { lat: 47.6062, lng: -122.3321, risk: 0.6 }, // Seattle area inland
    { lat: 47.4, lng: -121.8, risk: 0.65 },
    { lat: 47.8, lng: -121.5, risk: 0.7 },
  ];

  // Southwest US desert areas
  const southwestPoints = [
    { lat: 33.4484, lng: -112.0740, risk: 0.8 }, // Phoenix
    { lat: 33.6, lng: -111.8, risk: 0.75 },
    { lat: 33.2, lng: -112.3, risk: 0.7 },
    { lat: 35.6870, lng: -105.9378, risk: 0.7 }, // Santa Fe
    { lat: 35.9, lng: -105.7, risk: 0.65 },
    { lat: 35.4, lng: -106.1, risk: 0.75 },
    { lat: 39.5501, lng: -105.7821, risk: 0.75 }, // Colorado Rockies
    { lat: 39.8, lng: -105.5, risk: 0.7 },
    { lat: 39.3, lng: -106.0, risk: 0.8 },
  ];

  // Australia inland areas (avoiding coasts)
  const australiaPoints = [
    { lat: -33.6, lng: 150.8, risk: 0.8 }, // Sydney inland
    { lat: -33.4, lng: 150.6, risk: 0.75 },
    { lat: -33.8, lng: 150.9, risk: 0.7 },
    { lat: -37.6, lng: 144.6, risk: 0.7 }, // Melbourne inland
    { lat: -37.4, lng: 144.4, risk: 0.75 },
    { lat: -37.8, lng: 144.8, risk: 0.65 },
    { lat: -31.7, lng: 115.6, risk: 0.75 }, // Perth inland
    { lat: -31.5, lng: 115.4, risk: 0.7 },
    { lat: -31.9, lng: 115.8, risk: 0.8 },
  ];

  // Mediterranean Europe inland areas
  const mediterraneanPoints = [
    { lat: 40.4168, lng: -3.7038, risk: 0.6 }, // Madrid
    { lat: 40.6, lng: -3.5, risk: 0.55 },
    { lat: 40.2, lng: -3.9, risk: 0.65 },
    { lat: 37.9838, lng: 23.7275, risk: 0.65 }, // Athens
    { lat: 38.1, lng: 23.5, risk: 0.6 },
    { lat: 37.8, lng: 23.9, risk: 0.7 },
    { lat: 43.2965, lng: 5.3698, risk: 0.55 }, // Provence
    { lat: 43.5, lng: 5.1, risk: 0.5 },
    { lat: 43.0, lng: 5.6, risk: 0.6 },
  ];

  // Combine all points and add some variation (Claude)
  const allPoints = [...californiaPoints, ...pacificNWPoints, ...southwestPoints, ...australiaPoints, ...mediterraneanPoints];
  
  allPoints.forEach((point, index) => {
    // Add some nearby points with variation (Claude)
    for (let i = 0; i < 8; i++) {
      const latOffset = (Math.random() - 0.5) * 0.3; // ±0.15 degrees
      const lngOffset = (Math.random() - 0.5) * 0.3;
      const riskVariation = (Math.random() - 0.5) * 0.2; // ±0.1 risk variation
      
      const newLat = point.lat + latOffset;
      const newLng = point.lng + lngOffset;
      const newRisk = Math.max(0.1, Math.min(0.95, point.risk + riskVariation));
      
      wildfirePoints.push({
        id: `wildfire_${newLat.toFixed(4)}_${newLng.toFixed(4)}`,
        probability: parseFloat(newRisk.toFixed(3)),
        riskType: 'wildfire',
        lastAssessment: Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000
      });
    }
  });

  return {
    type: 'FeatureCollection',
    features: wildfirePoints.map(point => {
      const parts = point.id.split('_');
      const lat = parseFloat(parts[1]);
      const lng = parseFloat(parts[2]);
      
      return {
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [lng, lat]
        },
        properties: point
      };
    })
  };
};

// Generate realistic drought risk data (focused on arid/semi-arid land regions only)
const generateDroughtData = (): DisasterRiskGeojson => {
  const droughtPoints: DisasterRiskProps[] = [];
  
  // California Central Valley and inland desert areas
  const californiaPoints = [
    { lat: 36.7783, lng: -119.4179, risk: 0.85 }, // Central Valley
    { lat: 36.5, lng: -119.2, risk: 0.8 },
    { lat: 37.0, lng: -119.6, risk: 0.9 },
    { lat: 33.9425, lng: -117.2297, risk: 0.8 }, // Riverside inland
    { lat: 33.7, lng: -117.0, risk: 0.75 },
    { lat: 34.1, lng: -117.4, risk: 0.85 },
    { lat: 32.8, lng: -117.0, risk: 0.75 }, // San Diego inland
    { lat: 32.6, lng: -116.8, risk: 0.7 },
    { lat: 33.0, lng: -117.2, risk: 0.8 },
  ];

  // Southwest US desert regions
  const southwestPoints = [
    { lat: 33.4484, lng: -112.0740, risk: 0.9 }, // Phoenix/Arizona
    { lat: 33.6, lng: -111.8, risk: 0.85 },
    { lat: 33.2, lng: -112.3, risk: 0.95 },
    { lat: 35.0853, lng: -106.6056, risk: 0.85 }, // Albuquerque
    { lat: 35.3, lng: -106.4, risk: 0.8 },
    { lat: 34.8, lng: -106.8, risk: 0.9 },
    { lat: 31.7619, lng: -106.4850, risk: 0.8 }, // El Paso area
    { lat: 31.5, lng: -106.2, risk: 0.75 },
    { lat: 31.9, lng: -106.7, risk: 0.85 },
    { lat: 36.2, lng: -115.0, risk: 0.95 }, // Las Vegas inland
    { lat: 36.0, lng: -114.8, risk: 0.9 },
    { lat: 36.4, lng: -115.2, risk: 0.85 },
  ];

  // Great Plains (Dust Bowl region)
  const greatPlainsPoints = [
    { lat: 39.7391, lng: -104.9847, risk: 0.75 }, // Eastern Colorado
    { lat: 39.5, lng: -104.7, risk: 0.7 },
    { lat: 39.9, lng: -105.2, risk: 0.8 },
    { lat: 37.6872, lng: -97.3301, risk: 0.8 }, // Kansas
    { lat: 37.4, lng: -97.1, risk: 0.75 },
    { lat: 37.9, lng: -97.5, risk: 0.85 },
    { lat: 35.4676, lng: -97.5164, risk: 0.75 }, // Oklahoma
    { lat: 35.2, lng: -97.3, risk: 0.7 },
    { lat: 35.7, lng: -97.7, risk: 0.8 },
  ];

  // Australia's interior (far from coasts)
  const australiaPoints = [
    { lat: -30.7333, lng: 121.5000, risk: 0.9 }, // Western Australia interior
    { lat: -30.5, lng: 121.3, risk: 0.85 },
    { lat: -30.9, lng: 121.7, risk: 0.95 },
    { lat: -28.0167, lng: 153.0, risk: 0.85 }, // Queensland interior
    { lat: -27.8, lng: 152.8, risk: 0.8 },
    { lat: -28.2, lng: 153.2, risk: 0.9 },
    { lat: -32.5000, lng: 147.0000, risk: 0.8 }, // NSW interior
    { lat: -32.3, lng: 146.8, risk: 0.75 },
    { lat: -32.7, lng: 147.2, risk: 0.85 },
  ];

  // Southern Africa interior
  const africaPoints = [
    { lat: -25.7479, lng: 28.2293, risk: 0.85 }, // South Africa interior
    { lat: -25.5, lng: 28.0, risk: 0.8 },
    { lat: -25.9, lng: 28.4, risk: 0.9 },
    { lat: -22.9576, lng: 18.4904, risk: 0.8 }, // Namibia
    { lat: -22.7, lng: 18.2, risk: 0.75 },
    { lat: -23.1, lng: 18.7, risk: 0.85 },
  ];

  // Mediterranean interior (away from coasts)
  const mediterraneanPoints = [
    { lat: 40.4637, lng: -3.7492, risk: 0.65 }, // Central Spain
    { lat: 40.2, lng: -3.5, risk: 0.6 },
    { lat: 40.7, lng: -3.9, risk: 0.7 },
    { lat: 37.9755, lng: 23.7348, risk: 0.7 }, // Greece interior
    { lat: 37.7, lng: 23.5, risk: 0.65 },
    { lat: 38.1, lng: 23.9, risk: 0.75 },
    { lat: 36.7378, lng: 3.0867, risk: 0.75 }, // Algeria
    { lat: 36.5, lng: 2.8, risk: 0.7 },
    { lat: 36.9, lng: 3.3, risk: 0.8 },
  ];

  // Combine all points and add variation
  const allPoints = [...californiaPoints, ...southwestPoints, ...greatPlainsPoints, ...australiaPoints, ...africaPoints, ...mediterraneanPoints];
  
  allPoints.forEach((point, index) => {
    // Add some nearby points with variation
    for (let i = 0; i < 6; i++) {
      const latOffset = (Math.random() - 0.5) * 0.4; // ±0.2 degrees
      const lngOffset = (Math.random() - 0.5) * 0.4;
      const riskVariation = (Math.random() - 0.5) * 0.15; // ±0.075 risk variation
      
      const newLat = point.lat + latOffset;
      const newLng = point.lng + lngOffset;
      const newRisk = Math.max(0.1, Math.min(0.95, point.risk + riskVariation));
      
      droughtPoints.push({
        id: `drought_${newLat.toFixed(4)}_${newLng.toFixed(4)}`,
        probability: parseFloat(newRisk.toFixed(3)),
        riskType: 'drought',
        lastAssessment: Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000
      });
    }
  });

  return {
    type: 'FeatureCollection',
    features: droughtPoints.map(point => {
      const parts = point.id.split('_');
      const lat = parseFloat(parts[1]);
      const lng = parseFloat(parts[2]);
      
      return {
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [lng, lat]
        },
        properties: point
      };
    })
  };
};

// Main data loading functions
const loadWildfireData = (): Promise<DisasterRiskGeojson> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(generateWildfireData());
    }, 300); // Simulate API call delay
  });
};

const loadDroughtData = (): Promise<DisasterRiskGeojson> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(generateDroughtData());
    }, 300);
  });
};

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

// helper function to get color based on probability (0-1)
const getProbabilityColor = (probability: number, disasterType: string): string => {
  // Clamp probability to 0-1 range
  const p = Math.max(0, Math.min(1, probability));
  
  if (disasterType === 'wildfire') {
    // Red gradien for wildfires
    if (p >= 0.8) return '#8B0000'; // Dark red
    if (p >= 0.6) return '#FF0000'; // Red
    if (p >= 0.4) return '#FF4500'; // Orange-red
    if (p >= 0.2) return '#FF8C00'; // Orange
    return '#FFD700'; // Gold
  } else if (disasterType === 'drought') {
    // Brown gradient for drought
    if (p >= 0.8) return '#8B4513'; // Saddle brown
    if (p >= 0.6) return '#A0522D'; // Sienna
    if (p >= 0.4) return '#CD853F'; // Peru
    if (p >= 0.2) return '#DEB887'; // Burlywood
    return '#F5DEB3'; // Wheat
  }
  
  return '#808080'; // Gray in case of error
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
    } else if (disasterType === 'drought') {
      return [
        { range: '80-100%', color: '#8B4513', label: 'Very High' },
        { range: '60-80%', color: '#A0522D', label: 'High' },
        { range: '40-60%', color: '#CD853F', label: 'Moderate' },
        { range: '20-40%', color: '#DEB887', label: 'Low' },
        { range: '0-20%', color: '#F5DEB3', label: 'Very Low' },
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

//Page structure and effects
const GlobePage = () => {
  const [disasterData, setDisasterData] = useState<DisasterRiskGeojson>();
  const [use3D, setUse3D] = useState(true);
  const [selectedDisaster, setSelectedDisaster] = useState<any>(null);
  const [disasterType, setDisasterType] = useState<'wildfire' | 'drought'>('wildfire');
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
          case 'drought':
            data = await loadDroughtData();
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
            onChange={(e) => setDisasterType(e.target.value as 'wildfire' | 'drought')}
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
            <option value="drought">Droughts</option>
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