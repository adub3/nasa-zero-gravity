import React from 'react';
import {BrowserRouter as Router, Routes, Route} from 'react-router-dom';

import MethodsPage from './methods';
import AboutPage from './about';
import GlobePage from './globe';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<GlobePage />} />
        <Route path="/methods" element={<MethodsPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </Router>
  );
};

export default App;