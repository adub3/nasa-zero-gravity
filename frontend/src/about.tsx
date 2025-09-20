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
            About Us
          </h1>
          <p style={{
            fontSize: '1.2rem',
            color: '#666',
            maxWidth: '800px',
            margin: '0 auto'
          }}>
            We're a dedicated team of researchers and developers committed to making earthquake risk data accessible and actionable for communities worldwide.
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
          <h2 style={{ color: '#d32f2f', marginBottom: '2rem', fontSize: '2.5rem' }}>Our Mission</h2>
          <p style={{ 
            color: '#555', 
            fontSize: '1.3rem', 
            lineHeight: '1.7',
            maxWidth: '800px',
            margin: '0 auto'
          }}>
            To democratize access to earthquake risk information through innovative data visualization, 
            empowering individuals, communities, and organizations to make informed decisions about 
            seismic preparedness and risk mitigation.
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
              role="Lead Seismologist"
              bio="Ph.D. in Geophysics from Stanford University. 15+ years experience in earthquake hazard assessment and risk modeling. Previously worked with USGS Earthquake Hazards Program."
            />
            
            <TeamMember 
              name="Nicholas Copland"
              role="Full Stack Developer"
              bio="Expert in React, TypeScript, and geospatial data visualization. Specialized in creating intuitive interfaces for complex scientific data. Former lead developer at several GIS companies."
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
              role="Data Scientist"
              bio="Ph.D. in Applied Mathematics with focus on statistical modeling and machine learning. Develops algorithms for processing and analyzing large-scale seismic datasets."
            />
            
            <TeamMember 
              name="Tal Lucas"
              role="UX/UI Designer"
              bio="M.F.A. in Interactive Design from Art Center College. Specializes in data visualization and user experience design for scientific applications. Previously designed interfaces for NASA's Earth Science Division."
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
            <h3 style={{ color: '#d32f2f', marginBottom: '1.5rem', fontSize: '1.5rem' }}>Our Story</h3>
            <p style={{ color: '#555', marginBottom: '1rem' }}>
              Founded in 2023, our project began when our team recognized the gap between complex 
              seismic research data and public understanding. We saw communities making critical 
              decisions about where to live and work without access to clear, visual earthquake risk information.
            </p>
            <p style={{ color: '#555' }}>
              What started as a research collaboration has evolved into a comprehensive platform 
              that transforms raw seismic data into intuitive, actionable visualizations.
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
            <h3 style={{ color: '#d32f2f', marginBottom: '1.5rem', fontSize: '1.5rem' }}>Our Values</h3>
            <ul style={{ color: '#555', paddingLeft: '1.2rem', listStyle: 'none' }}>
              <li style={{ marginBottom: '0.8rem', position: 'relative', paddingLeft: '1.5rem' }}>
                <span style={{ position: 'absolute', left: 0, color: '#d32f2f', fontWeight: 'bold' }}>•</span>
                <strong>Transparency:</strong> Open methodology and clear data sources
              </li>
              <li style={{ marginBottom: '0.8rem', position: 'relative', paddingLeft: '1.5rem' }}>
                <span style={{ position: 'absolute', left: 0, color: '#d32f2f', fontWeight: 'bold' }}>•</span>
                <strong>Accessibility:</strong> Making complex data understandable for everyone
              </li>
              <li style={{ marginBottom: '0.8rem', position: 'relative', paddingLeft: '1.5rem' }}>
                <span style={{ position: 'absolute', left: 0, color: '#d32f2f', fontWeight: 'bold' }}>•</span>
                <strong>Accuracy:</strong> Rigorous scientific standards in all our work
              </li>
              <li style={{ marginBottom: '0.8rem', position: 'relative', paddingLeft: '1.5rem' }}>
                <span style={{ position: 'absolute', left: 0, color: '#d32f2f', fontWeight: 'bold' }}>•</span>
                <strong>Impact:</strong> Helping communities become more resilient
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
          <h3 style={{ color: '#d32f2f', marginBottom: '1.5rem', fontSize: '2rem' }}>Get In Touch</h3>
          <p style={{ color: '#555', marginBottom: '2rem', fontSize: '1.1rem' }}>
            Have questions about our methodology? Interested in collaborating? We'd love to hear from you.
          </p>
          
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
              <strong style={{ color: '#d32f2f' }}>Email:</strong>
              <br />
              <span style={{ color: '#555' }}>contact@earthquakerisk.org</span>
            </div>
            
            <div style={{
              padding: '1rem 2rem',
              background: 'rgba(211, 47, 47, 0.1)',
              borderRadius: '12px',
              border: '1px solid rgba(211, 47, 47, 0.2)'
            }}>
              <strong style={{ color: '#d32f2f' }}>GitHub:</strong>
              <br />
              <span style={{ color: '#555' }}>github.com/earthquakerisk</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default AboutPage;