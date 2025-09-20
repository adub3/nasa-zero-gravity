import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const NavigationBar = () => {
  const location = useLocation();
  
  const linkStyle = (path: string) => ({
    color: location.pathname === path ? '#d32f2f' : '#333',
    textDecoration: 'none',
    fontWeight: location.pathname === path ? '600' : '500',
    transition: 'color 0.3s ease'
  });

  return (
    <nav style={{
      position: 'fixed',
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
      <Link to="/" style={{
        fontSize: '1.5rem',
        fontWeight: 'bold',
        color: '#333',
        textDecoration: 'none'
      }}>
        Earthquake Risk Map
      </Link>
      
      <div style={{
        display: 'flex',
        gap: '2rem'
      }}>
        <Link to="/" style={linkStyle('/')}>
          Map
        </Link>
        <Link to="/methods" style={linkStyle('/methods')}>
          Methods
        </Link>
        <Link to="/about" style={linkStyle('/about')}>
          About Us
        </Link>
      </div>
    </nav>
  );
};

export default NavigationBar;