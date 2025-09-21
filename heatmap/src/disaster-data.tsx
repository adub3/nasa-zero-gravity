//Claude Generating data for our UI and demo
import { FeatureCollection, Point } from 'geojson';

export type DisasterRiskProps = {
  id: string;
  probability: number; // 0-1 probability value
  riskType: 'wildfire' | 'flood' | 'hurricane';
  lastAssessment: number; // timestamp
};

export type DisasterRiskGeojson = FeatureCollection<Point, DisasterRiskProps>;

// Generate grid-based risk data around specific hotspots
const generateRiskGrid = (
  centerLat: number,
  centerLng: number,
  gridSize: number,
  spacing: number,
  riskType: 'wildfire' | 'flood' | 'hurricane',
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

// Generate wildfire risk data (focused on western US, Australia, etc.)
const generateWildfireData = (): DisasterRiskGeojson => {
  const wildfirePoints: DisasterRiskProps[] = [];
  
  // California hotspots
  wildfirePoints.push(...generateRiskGrid(34.0522, -118.2437, 15, 0.1, 'wildfire', 0.7)); // LA area
  wildfirePoints.push(...generateRiskGrid(37.7749, -122.4194, 12, 0.08, 'wildfire', 0.8)); // Bay Area
  
  // Pacific Northwest
  wildfirePoints.push(...generateRiskGrid(45.5152, -122.6784, 10, 0.12, 'wildfire', 0.6)); // Portland
  
  // Australia
  wildfirePoints.push(...generateRiskGrid(-33.8688, 151.2093, 12, 0.1, 'wildfire', 0.75)); // Sydney
  wildfirePoints.push(...generateRiskGrid(-37.8136, 144.9631, 10, 0.08, 'wildfire', 0.65)); // Melbourne
  
  // Mediterranean (Spain, Greece)
  wildfirePoints.push(...generateRiskGrid(40.4168, -3.7038, 8, 0.15, 'wildfire', 0.55)); // Madrid
  wildfirePoints.push(...generateRiskGrid(37.9838, 23.7275, 8, 0.12, 'wildfire', 0.6)); // Athens
  
  return {
    type: 'FeatureCollection',
    features: wildfirePoints.map((point, index) => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [
          parseFloat((Math.random() * 360 - 180).toFixed(4)), // Random lng for now
          parseFloat((Math.random() * 180 - 90).toFixed(4))   // Random lat for now
        ]
      },
      properties: point
    }))
  };
};

// Generate flood risk data (focused on coastal areas, river basins)
const generateFloodData = (): DisasterRiskGeojson => {
  const floodPoints: DisasterRiskProps[] = [];
  
  // US Gulf Coast
  floodPoints.push(...generateRiskGrid(29.7604, -95.3698, 12, 0.08, 'flood', 0.8)); // Houston
  floodPoints.push(...generateRiskGrid(29.9511, -90.0715, 10, 0.1, 'flood', 0.85)); // New Orleans
  
  // US East Coast
  floodPoints.push(...generateRiskGrid(40.7128, -74.0060, 14, 0.06, 'flood', 0.7)); // NYC
  floodPoints.push(...generateRiskGrid(25.7617, -80.1918, 10, 0.08, 'flood', 0.75)); // Miami
  
  // European rivers
  floodPoints.push(...generateRiskGrid(51.5074, -0.1278, 8, 0.12, 'flood', 0.6)); // London (Thames)
  floodPoints.push(...generateRiskGrid(52.3676, 4.9041, 10, 0.08, 'flood', 0.8)); // Amsterdam
  
  // Asian deltas
  floodPoints.push(...generateRiskGrid(10.8231, 106.6297, 12, 0.1, 'flood', 0.85)); // Ho Chi Minh City
  floodPoints.push(...generateRiskGrid(23.8103, 90.4125, 10, 0.15, 'flood', 0.9)); // Dhaka
  
  return {
    type: 'FeatureCollection',
    features: floodPoints.map((point, index) => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [
          parseFloat((Math.random() * 360 - 180).toFixed(4)),
          parseFloat((Math.random() * 180 - 90).toFixed(4))
        ]
      },
      properties: point
    }))
  };
};

// Generate hurricane risk data (focused on hurricane-prone regions)
const generateHurricaneData = (): DisasterRiskGeojson => {
  const hurricanePoints: DisasterRiskProps[] = [];
  
  // US Atlantic Coast
  hurricanePoints.push(...generateRiskGrid(25.7617, -80.1918, 14, 0.08, 'hurricane', 0.8)); // Miami
  hurricanePoints.push(...generateRiskGrid(32.0835, -81.0998, 10, 0.1, 'hurricane', 0.7)); // Savannah
  hurricanePoints.push(...generateRiskGrid(35.7796, -75.9442, 8, 0.12, 'hurricane', 0.65)); // Outer Banks
  
  // Gulf of Mexico
  hurricanePoints.push(...generateRiskGrid(29.7604, -95.3698, 12, 0.1, 'hurricane', 0.75)); // Houston
  hurricanePoints.push(...generateRiskGrid(30.6944, -88.0431, 8, 0.15, 'hurricane', 0.7)); // Mobile Bay
  
  // Caribbean
  hurricanePoints.push(...generateRiskGrid(18.2208, -66.5901, 10, 0.08, 'hurricane', 0.85)); // Puerto Rico
  hurricanePoints.push(...generateRiskGrid(25.0343, -77.3963, 8, 0.12, 'hurricane', 0.8)); // Bahamas
  
  // Pacific (typhoons)
  hurricanePoints.push(...generateRiskGrid(14.5995, 120.9842, 12, 0.1, 'hurricane', 0.9)); // Philippines
  hurricanePoints.push(...generateRiskGrid(26.2041, 127.6793, 10, 0.08, 'hurricane', 0.75)); // Okinawa
  
  return {
    type: 'FeatureCollection',
    features: hurricanePoints.map((point, index) => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [
          parseFloat((Math.random() * 360 - 180).toFixed(4)),
          parseFloat((Math.random() * 180 - 90).toFixed(4))
        ]
      },
      properties: point
    }))
  };
};

// Main data loading functions
export const loadWildfireData = (): Promise<DisasterRiskGeojson> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(generateWildfireData());
    }, 300); // Simulate API call delay
  });
};

export const loadFloodData = (): Promise<DisasterRiskGeojson> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(generateFloodData());
    }, 300);
  });
};

export const loadHurricaneData = (): Promise<DisasterRiskGeojson> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(generateHurricaneData());
    }, 300);
  });
};