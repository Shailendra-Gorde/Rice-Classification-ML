import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Box,
  Drawer,
  AppBar,
  Toolbar,
  List,
  Typography,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  CircularProgress,
  Container
} from '@mui/material';
import {
  Dataset as DatasetIcon,
  Analytics as AnalyticsIcon,
  Science as ScienceIcon,
  Image as ImageIcon,
  History as HistoryIcon,
  Store as StoreIcon,
  CompareArrows as CompareIcon,
  AdminPanelSettings as AdminIcon
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import DatasetPage from './pages/DatasetPage';
import AnalysisPage from './pages/AnalysisPage';
import PredictPage from './pages/PredictPage';
import HistoryPage from './pages/HistoryPage';
import MarketplacePage from './pages/MarketplacePage';
import ComparePage from './pages/ComparePage';
import AdminPage from './pages/AdminPage';

const drawerWidth = 260;

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#0a1929',
      paper: '#1a2332',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'linear-gradient(rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.05))',
        },
      },
    },
  },
});

// Helper function to get page title based on current path
const getPageTitle = (pathname) => {
  if (pathname === '/predict') {
    return 'Rice Type Classification - Predict';
  } else if (pathname === '/history') {
    return 'Rice Type Classification - History';
  } else if (pathname === '/dataset' || pathname === '/') {
    return 'Rice Type Classification - Dataset';
  } else if (pathname === '/analysis') {
    return 'Rice Type Classification - Analysis';
  } else if (pathname === '/marketplace') {
    return 'Rice Type Classification - Marketplace';
  } else if (pathname === '/compare') {
    return 'Rice Type Classification - Compare';
  } else if (pathname === '/admin') {
    return 'Rice Type Classification - Admin';
  }
  return 'Rice Type Classification - ML Dashboard';
};

// Inner component that uses useLocation
function DashboardContent({ dashboardData, onDataRefresh }) {
  const location = useLocation();

  const menuItems = [
    { text: 'Predict', icon: <ImageIcon />, path: '/predict', page: 'predict' },
    { text: 'Compare', icon: <CompareIcon />, path: '/compare', page: 'compare' },
    { text: 'History', icon: <HistoryIcon />, path: '/history', page: 'history' },
    { text: 'Marketplace', icon: <StoreIcon />, path: '/marketplace', page: 'marketplace' },
    { text: 'Dataset', icon: <DatasetIcon />, path: '/dataset', page: 'dataset' },
    { text: 'Analysis', icon: <AnalyticsIcon />, path: '/analysis', page: 'analysis' },
    { text: 'Admin', icon: <AdminIcon />, path: '/admin', page: 'admin' },
  ];

  // Get dynamic page title based on current route
  const pageTitle = getPageTitle(location.pathname);

  // Check if current menu item is selected based on location
  const isSelected = (itemPath) => {
    return location.pathname === itemPath || (location.pathname === '/' && itemPath === '/dataset');
  };

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{
          width: `calc(100% - ${drawerWidth}px)`,
          ml: `${drawerWidth}px`,
          backgroundColor: '#1a2332',
        }}
      >
        <Toolbar>
          <ScienceIcon sx={{ mr: 2 }} />
          <Typography variant="h6" noWrap component="div">
            {pageTitle}
          </Typography>
        </Toolbar>
      </AppBar>

          <Drawer
            sx={{
              width: drawerWidth,
              flexShrink: 0,
              '& .MuiDrawer-paper': {
                width: drawerWidth,
                boxSizing: 'border-box',
                backgroundColor: '#0d1b2a',
              },
            }}
            variant="permanent"
            anchor="left"
          >
            <Toolbar>
              {/* <Typography variant="h6" sx={{ fontWeight: 'bold', color: '#90caf9' }}>
                Navigation
              </Typography> */}
            </Toolbar>
            <List>
              {menuItems.map((item) => (
                <ListItem key={item.text} disablePadding>
                  <ListItemButton
                    selected={isSelected(item.path)}
                    component="a"
                    href={item.path}
                    sx={{
                      '&.Mui-selected': {
                        backgroundColor: 'rgba(144, 202, 249, 0.16)',
                        '&:hover': {
                          backgroundColor: 'rgba(144, 202, 249, 0.24)',
                        },
                      },
                    }}
                  >
                    <ListItemIcon sx={{ color: isSelected(item.path) ? '#90caf9' : 'inherit' }}>
                      {item.icon}
                    </ListItemIcon>
                    <ListItemText primary={item.text} />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          </Drawer>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          minHeight: '100vh',
        }}
      >
        <Toolbar />
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Container maxWidth="xl">
            <Routes>
              <Route path="/" element={<Navigate to="/predict" replace />} />
              <Route path="/predict" element={<PredictPage />} />
              <Route path="/history" element={<HistoryPage />} />
              <Route path="/marketplace" element={<MarketplacePage />} />
              <Route path="/compare" element={<ComparePage />} />
              <Route path="/admin" element={<AdminPage />} />
              <Route path="/dataset" element={<DatasetPage data={dashboardData} />} />
              <Route path="/analysis" element={<AnalysisPage data={dashboardData} onDataRefresh={onDataRefresh} />} />
            </Routes>
          </Container>
        </motion.div>
      </Box>
    </Box>
  );
}

function App() {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);

  const loadDashboardData = () => {
    // Try to load from API first (dynamic data), fallback to static JSON
    fetch('http://localhost:5001/api/dashboard-data')
      .then(response => {
        if (!response.ok) throw new Error('API not available');
        return response.json();
      })
      .then(apiData => {
        if (apiData.success && apiData.data) {
          // Merge API data with static data
          fetch('/dashboard_data.json')
            .then(response => response.json())
            .then(staticData => {
              // Merge: use API data for dynamic parts, static for structure
              const merged = {
                ...staticData,
                project: {
                  ...staticData.project,
                  total_samples: apiData.data.project.total_samples || staticData.project.total_samples || 0,
                  num_features: apiData.data.project.num_features || staticData.project.num_features,
                  num_classes: apiData.data.project.num_classes || staticData.project.num_classes,
                  class_names: apiData.data.project.class_names || staticData.project.class_names || []
                },
                dataset: {
                  ...staticData.dataset,
                  class_distribution: apiData.data.dataset.class_distribution || staticData.dataset.class_distribution || {},
                  statistics: {
                    ...staticData.dataset.statistics,
                    ...(apiData.data.dataset.statistics || {})
                  },
                  imbalance_ratio: apiData.data.dataset.imbalance_ratio || staticData.dataset.imbalance_ratio || 1.0
                },
                // IMPORTANT: Use API data for models (these have the real values from metadata)
                best_model: apiData.data.best_model || staticData.best_model,
                all_models: apiData.data.all_models || staticData.all_models || [],
                model_rankings: apiData.data.model_rankings || staticData.model_rankings || [],
                prediction_history: apiData.data.prediction_history || { total: 0, recent_predictions: [] }
              };
              console.log('[App.js] Merged data - best_model:', merged.best_model);
              console.log('[App.js] Merged data - all_models:', merged.all_models);
              setDashboardData(merged);
              if (loading) setLoading(false);
            })
            .catch(() => {
              // If static load fails, use API data with defaults
              const apiDataWithDefaults = {
                ...apiData.data,
                dataset: {
                  ...apiData.data.dataset,
                  features: {
                    all_features: [],
                    color_features: [],
                    texture_features: [],
                    size_features: [],
                    shape_features: [],
                    feature_descriptions: {}
                  }
                }
              };
              console.log('[App.js] Using API data only - best_model:', apiDataWithDefaults.best_model);
              console.log('[App.js] Using API data only - all_models:', apiDataWithDefaults.all_models);
              setDashboardData(apiDataWithDefaults);
              if (loading) setLoading(false);
            });
        } else {
          // Fallback to static data
          fetch('/dashboard_data.json')
            .then(response => response.json())
            .then(data => {
              setDashboardData(data);
              if (loading) setLoading(false);
            })
            .catch(error => {
              console.error('Error loading dashboard data:', error);
              if (loading) setLoading(false);
            });
        }
      })
      .catch(() => {
        // API not available, use static data
        fetch('/dashboard_data.json')
          .then(response => response.json())
          .then(data => {
            setDashboardData(data);
            if (loading) setLoading(false);
          })
          .catch(error => {
            console.error('Error loading dashboard data:', error);
            if (loading) setLoading(false);
          });
      });
  };

  useEffect(() => {
    loadDashboardData();
    
    // Refresh dashboard data every 5 seconds
    const interval = setInterval(loadDashboardData, 5000);
    
    // Listen for prediction events to refresh immediately
    const handlePrediction = () => {
      loadDashboardData();
    };
    window.addEventListener('predictionMade', handlePrediction);
    
    return () => {
      clearInterval(interval);
      window.removeEventListener('predictionMade', handlePrediction);
    };
  }, []);

  if (loading) {
    return (
      <ThemeProvider theme={darkTheme}>
        <CssBaseline />
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: '100vh',
            flexDirection: 'column',
            gap: 2
          }}
        >
          <CircularProgress size={60} />
          <Typography variant="h6">Loading Rice ML Dashboard...</Typography>
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Router>
        <DashboardContent dashboardData={dashboardData} onDataRefresh={loadDashboardData} />
      </Router>
    </ThemeProvider>
  );
}

export default App;
