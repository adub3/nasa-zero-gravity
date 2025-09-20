import React from 'react';

const HeatmapKey = () => {
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
        Earthquake Risk
      </h4>
      
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem'
      }}>
        <div style={{
          height: '20px',
          background: 'linear-gradient(to right, rgba(255, 0, 0, 0), rgba(255, 0, 0, 1))',
          borderRadius: '10px',
          border: '1px solid rgba(0, 0, 0, 0.1)'
        }} />
        
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: '0.875rem',
          color: '#666',
          marginTop: '0.5rem'
        }}>
          <span>Low</span>
          <span>High</span>
        </div>
      </div>
      
      <div style={{
        marginTop: '1rem',
        fontSize: '0.75rem',
        color: '#888',
        lineHeight: '1.4'
      }}>
        Risk levels based on earthquake magnitude and frequency data
      </div>
    </div>
  );
};

export default HeatmapKey;