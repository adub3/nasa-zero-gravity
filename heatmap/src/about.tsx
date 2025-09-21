import React from 'react';
import NavigationBar from './navigation-bar';

type TeamMemberProps = {
    name: string;
    role: string;
    bio: string;
    image?: string; // optional now
  };
  

const TeamMember: React.FC<TeamMemberProps> = ({ name, role, bio, image }) =>  (
  <div style={{
    background: 'rgba(255, 255, 255, 0.9)',
    backdropFilter: 'blur(10px)',
    borderRadius: '16px',
    padding: '2rem',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
    border: '1px solid rgba(255, 255, 255, 0.2)',
    textAlign: 'center',
    transition: 'transform 0.3s ease'
  }}
  onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-5px)'}
  onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0px)'}
  >
    <div style={{
      width: '120px',
      height: '120px',
      borderRadius: '50%',
      background: image || 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      margin: '0 auto 1.5rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '2.5rem',
      fontWeight: 'bold',
      color: 'white'
    }}>
      {!image && name.charAt(0)}
    </div>
    <h3 style={{ color: '#333', marginBottom: '0.5rem', fontSize: '1.5rem' }}>{name}</h3>
    <p style={{ color: '#d32f2f', fontWeight: '600', marginBottom: '1rem', fontSize: '1rem' }}>{role}</p>
    <p style={{ color: '#666', lineHeight: '1.5', fontSize: '0.95rem' }}>{bio}</p>
  </div>
);

const AboutPage = () => {
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
        
        <header style={{ textAlign: 'center', marginBottom: '4rem' }}>
          <h1 style={{
            fontSize: '3rem',
            fontWeight: '700',
            color: '#333',
            marginBottom: '1rem'
          }}>
            About Our Team
          </h1>
          <p style={{
            fontSize: '1.2rem',
            color: '#666',
            maxWidth: '800px',
            margin: '0 auto'
          }}>
            We're a team of UNC students who developed this natural disaster risk visualization platform for the CDC Hackathon, combining NASA weather data with machine learning to help communities prepare for climate-related disasters.
          </p>
        </header>

        <div style={{
          background: 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(10px)',
          borderRadius: '16px',
          padding: '3rem',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          marginBottom: '4rem',
          textAlign: 'center'
        }}>
          <h2 style={{ color: '#d32f2f', marginBottom: '2rem', fontSize: '2.5rem' }}>CDC Hackathon at UNC</h2>
          <p style={{ 
            color: '#555', 
            fontSize: '1.3rem', 
            lineHeight: '1.7',
            maxWidth: '800px',
            margin: '0 auto'
          }}>
            This project was created during the CDC Hackathon held at the University of North Carolina. 
            Our goal was to leverage NASA's satellite weather data to create an accessible, real-time disaster 
            risk visualization tool that can help communities and public health officials make informed decisions 
            about natural disaster preparedness and response.
          </p>
        </div>

        <div style={{ marginBottom: '4rem' }}>
          <h2 style={{ 
            textAlign: 'center', 
            color: '#333', 
            marginBottom: '3rem',
            fontSize: '2.5rem'
          }}>
            Meet Our Team
          </h2>
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '2rem',
            marginBottom: '3rem'
          }}>
            
            <TeamMember 
              name="Andrew Wang"
              role="Predictive Modeling & Data Science"
              bio="UNC Statistics student. Andrew developed the core risk prediction models that analyze NASA satellite data to forecast disaster probabilities using neural networks and statistical modeling techniques."
            />
            
            <TeamMember 
              name="Nicholas Copland"
              role="Frontend Developer & UI/UX"
              bio="UNC Statistics student with expertise in React and modern web technologies. Nicholas designed and built the entire user interface, creating the interactive 3D globe visualization and mapping the data over it."
            />

          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '2rem',
            marginBottom: '3rem'
          }}>
            <TeamMember 
              name="Nishil Patel"
              role="Full Stack Developer & Systems Architecture"
              bio="UNC Statistics student. Nishil worked on backend data cleaning and the backend to frontend pipeline."
            />
            
            <TeamMember 
              name="Tal Lucas"
              role="Data Analytics & Backend Engineering"
              bio="UNC Physics student. Tal developed the data processing infrastructure that ingests and analyzes massive NASA weather datasets, implementing algorithms to indentify and extract meaningful risk indicators from satellite imagery."
            />
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
              We recognized that wildfires and drought due to global warming increasingly pose significant health risks
              to communities. Keeping with the theme we wanted to leverage NASA's resources to predict them. 
            </p>
            <p style={{ color: '#555' }}>
              Our team decided to focus on predictive visualization, giving probabilistic context to the weather patterns captured by 
              NASA satellites.
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
                <strong>NASA Data Integration:</strong> Working with real satellite weather data at scale
              </li>
              <li style={{ marginBottom: '0.8rem', position: 'relative', paddingLeft: '1.5rem' }}>
                <span style={{ position: 'absolute', left: 0, color: '#d32f2f', fontWeight: 'bold' }}>•</span>
                <strong>Machine Learning:</strong> Applying predictive models to climate and disaster data
              </li>
              <li style={{ marginBottom: '0.8rem', position: 'relative', paddingLeft: '1.5rem' }}>
                <span style={{ position: 'absolute', left: 0, color: '#d32f2f', fontWeight: 'bold' }}>•</span>
                <strong>Data Visualization:</strong> Making complex scientific data accessible to the public
              </li>
              <li style={{ marginBottom: '0.8rem', position: 'relative', paddingLeft: '1.5rem' }}>
                <span style={{ position: 'absolute', left: 0, color: '#d32f2f', fontWeight: 'bold' }}>•</span>
                <strong>Impact:</strong> Example of how technology can save lives
              </li>
            </ul>
          </div>

        </div>

        <div style={{
          background: 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(10px)',
          borderRadius: '16px',
          padding: '2.5rem',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          textAlign: 'center'
        }}>
          <h3 style={{ color: '#d32f2f', marginBottom: '1.5rem', fontSize: '2rem' }}>Our Code</h3>
          
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '2rem',
            flexWrap: 'wrap'
          }}>
            <div style={{
              padding: '1rem 2rem',
              background: 'rgba(211, 47, 47, 0.1)',
              borderRadius: '12px',
              border: '1px solid rgba(211, 47, 47, 0.2)'
            }}>
              <strong style={{ color: '#d32f2f' }}>Demo:</strong>
              <br />
              <span style={{ color: '#555' }}>Coming soon...</span>
            </div>
            
            <div style={{
              padding: '1rem 2rem',
              background: 'rgba(211, 47, 47, 0.1)',
              borderRadius: '12px',
              border: '1px solid rgba(211, 47, 47, 0.2)'
            }}>
              <strong style={{ color: '#d32f2f' }}>Github:</strong>
              <br />
              <span style={{ color: '#555' }}>https://github.com/adub3/nasa-zero-gravity</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default AboutPage;