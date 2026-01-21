import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Alert,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  History as HistoryIcon,
  Image as ImageIcon,
  Info as InfoIcon,
  Close as CloseIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';
import { motion } from 'framer-motion';

const API_BASE_URL = 'http://localhost:5001/api';

function HistoryPage() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [error, setError] = useState(null);
  const [clearing, setClearing] = useState(false);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/prediction-history?limit=100`);
      const data = await response.json();
      
      if (data.success) {
        setPredictions(data.predictions || []);
      } else {
        setError('Failed to load prediction history');
      }
    } catch (err) {
      setError(`Error loading history: ${err.message}. Make sure API server is running.`);
      console.error('History load error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
    
    // Refresh every 10 seconds
    const interval = setInterval(loadHistory, 10000);
    
    // Listen for new predictions
    const handlePrediction = () => {
      console.log('HistoryPage: Prediction made, refreshing history...');
      // Wait a bit longer for backend to save
      setTimeout(() => {
        loadHistory();
      }, 2000);
    };
    window.addEventListener('predictionMade', handlePrediction);
    
    return () => {
      clearInterval(interval);
      window.removeEventListener('predictionMade', handlePrediction);
    };
  }, []);

  const handleImageClick = (prediction) => {
    setSelectedPrediction(prediction);
    setDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setDialogOpen(false);
    setSelectedPrediction(null);
  };

  const handleClearHistory = async () => {
    if (!window.confirm('Are you sure you want to clear all prediction history? This action cannot be undone.')) {
      return;
    }

    setClearing(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/clear-history`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      const data = await response.json();
      
      if (data.success) {
        setPredictions([]);
        // Reload history to get updated state
        await loadHistory();
      } else {
        setError(data.error || 'Failed to clear history');
      }
    } catch (err) {
      setError(`Error clearing history: ${err.message}`);
      console.error('Clear history error:', err);
    } finally {
      setClearing(false);
    }
  };

  const getImageUrl = (imagePath) => {
    if (!imagePath) {
      console.log('getImageUrl: No imagePath provided');
      return null;
    }
    
    // If already a full URL, return as is
    if (imagePath.startsWith('http://') || imagePath.startsWith('https://')) {
      return imagePath;
    }
    
    // Check if it's a public folder path (prediction_images)
    if (imagePath.includes('prediction_images') || imagePath.startsWith('/prediction_images/')) {
      // Use public folder path directly (served by React dev server)
      let path = imagePath;
      if (!path.startsWith('/')) {
        path = `/${path}`;
      }
      // Remove /api/images/ prefix if present
      path = path.replace('/api/images/', '/prediction_images/');
      console.log('Image URL (public folder):', path);
      return path;
    }
    
    // Fallback to API endpoint
    let path = imagePath;
    if (!path.startsWith('/')) {
      path = `/${path}`;
    }
    
    // If path doesn't start with /api/images, add it
    if (!path.startsWith('/api/images/')) {
      // Extract filename from path
      const filename = path.split('/').pop();
      path = `/api/images/${filename}`;
    }
    
    const url = `${API_BASE_URL}${path}`;
    console.log('Image URL constructed:', { original: imagePath, path, url });
    return url;
  };

  const formatDate = (timestamp) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch {
      return timestamp;
    }
  };

  // Feature categories as requested
  const featureCategories = {
    'Color': {
      'Mean R (Red)': 'Color_Mean_R',
      'Mean G (Green)': 'Color_Mean_G',
      'Mean B (Blue)': 'Color_Mean_B',
      'Mean H (Hue)': 'Color_Mean_H',
      'Mean S (Saturation)': 'Color_Mean_S',
      'Mean V (Value)': 'Color_Mean_V'
    },
    'Texture': {
      'Contrast': 'Texture_Contrast',
      'Dissimilarity': 'Texture_Dissimilarity',
      'Homogeneity': 'Texture_Homogeneity',
      'Energy': 'Texture_Energy',
      'LBP Mean': 'Texture_LBP_Mean',
      'LBP Std': 'Texture_LBP_Std'
    },
    'Size': {
      'Area': 'Size_Area',
      'Perimeter': 'Size_Perimeter'
    },
    'Shape': {
      'Major Axis Length': 'Shape_MajorAxisLength',
      'Minor Axis Length': 'Shape_MinorAxisLength',
      'Convex Area': 'Shape_ConvexArea',
      'Eccentricity': 'Shape_Eccentricity',
      'Extent': 'Shape_Extent',
      'Roundness': 'Shape_Roundness',
      'Aspect Ratio': 'Shape_AspectRatio',
      'Equivalent Diameter': 'Shape_EquivDiameter',
      'Solidity': 'Shape_Solidity'
    }
  };

  if (loading && predictions.length === 0) {
    return (
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
          <CircularProgress size={60} />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
          <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold' }}>
            <HistoryIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Prediction History
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Tooltip title="Refresh">
              <IconButton onClick={loadHistory} color="primary">
                <RefreshIcon />
              </IconButton>
            </Tooltip>
            {predictions.length > 0 && (
              <Button
                variant="outlined"
                color="error"
                startIcon={clearing ? <CircularProgress size={20} /> : <DeleteIcon />}
                onClick={handleClearHistory}
                disabled={clearing}
                size="small"
              >
                {clearing ? 'Clearing...' : 'Clear History'}
              </Button>
            )}
            <Chip 
              label={`${predictions.length} predictions`} 
              color="primary"
            />
          </Box>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {predictions.length === 0 ? (
          <Card>
            <CardContent sx={{ textAlign: 'center', py: 8 }}>
              <ImageIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No predictions yet
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Upload images and make predictions to see them here
              </Typography>
            </CardContent>
          </Card>
        ) : (
          <Grid container spacing={3}>
            {predictions.map((prediction) => {
              // Use image_path if available, otherwise construct from image_filename
              let imagePath = null;
              if (prediction.image_path) {
                imagePath = prediction.image_path;
              } else if (prediction.image_filename) {
                imagePath = `/api/images/${prediction.image_filename}`;
              }
              
              const imageUrl = imagePath ? getImageUrl(imagePath) : null;
              
              // Debug logging for first prediction
              if (prediction.id === (predictions[0]?.id || predictions[0]?.id)) {
                console.log('Prediction image data:', {
                  id: prediction.id,
                  image_path: prediction.image_path,
                  image_filename: prediction.image_filename,
                  constructed_path: imagePath,
                  final_url: imageUrl
                });
              }
              
              return (
                <Grid item xs={12} sm={6} md={4} lg={3} key={prediction.id}>
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <Card 
                      sx={{ 
                        cursor: 'pointer',
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column'
                      }}
                      onClick={() => handleImageClick(prediction)}
                    >
                      <Box
                        sx={{
                          width: '100%',
                          height: 200,
                          backgroundColor: 'background.default',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          overflow: 'hidden'
                        }}
                      >
                        {imageUrl ? (
                          <Box
                            component="img"
                            src={imageUrl}
                            alt={prediction.variety}
                            sx={{
                              width: '100%',
                              height: '100%',
                              objectFit: 'contain'
                            }}
                            onError={(e) => {
                              console.error('Image failed to load:', imageUrl, prediction);
                              e.target.style.display = 'none';
                              if (e.target.nextSibling) {
                                e.target.nextSibling.style.display = 'flex';
                              }
                            }}
                            onLoad={() => {
                              console.log('Image loaded successfully:', imageUrl);
                            }}
                          />
                        ) : (
                          <Box
                            sx={{
                              display: 'flex',
                              flexDirection: 'column',
                              alignItems: 'center',
                              color: 'text.secondary'
                            }}
                          >
                            <ImageIcon sx={{ fontSize: 48, mb: 1 }} />
                            <Typography variant="caption">No Image Available</Typography>
                            {prediction.image_filename && (
                              <Typography variant="caption" color="error" sx={{ mt: 1 }}>
                                Filename: {prediction.image_filename}
                              </Typography>
                            )}
                          </Box>
                        )}
                      </Box>
                      <CardContent sx={{ flexGrow: 1 }}>
                        <Chip
                          label={prediction.variety || 'Unknown'}
                          color={prediction.variety && prediction.variety !== 'Unknown' ? 'primary' : 'default'}
                          sx={{ mb: 1, fontWeight: 'bold' }}
                        />
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Confidence: {prediction.confidence?.toFixed(2) || '0.00'}%
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {formatDate(prediction.timestamp)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </motion.div>
                </Grid>
              );
            })}
          </Grid>
        )}

        {/* Detail Dialog */}
        <Dialog
          open={dialogOpen}
          onClose={handleCloseDialog}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="h6">
                Prediction Details
              </Typography>
              <IconButton onClick={handleCloseDialog}>
                <CloseIcon />
              </IconButton>
            </Box>
          </DialogTitle>
          <DialogContent>
            {selectedPrediction && (
              <Box>
                {/* Image */}
                {(() => {
                  let imagePath = null;
                  if (selectedPrediction.image_path) {
                    imagePath = selectedPrediction.image_path;
                  } else if (selectedPrediction.image_filename) {
                    imagePath = `/api/images/${selectedPrediction.image_filename}`;
                  }
                  const imageUrl = imagePath ? getImageUrl(imagePath) : null;
                  console.log('Dialog image URL:', { imagePath, imageUrl, prediction: selectedPrediction });
                  return imageUrl ? (
                    <Box sx={{ mb: 3, textAlign: 'center' }}>
                      <Box
                        component="img"
                        src={imageUrl}
                        alt={selectedPrediction.variety}
                      sx={{
                        maxWidth: '100%',
                        maxHeight: '300px',
                        borderRadius: 1,
                        objectFit: 'contain'
                      }}
                      onError={(e) => {
                        console.error('Image load error:', imageUrl);
                        e.target.style.display = 'none';
                      }}
                    />
                  </Box>
                  ) : null;
                })()}

                {/* Prediction Info */}
                <Grid container spacing={2} sx={{ mb: 3 }}>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2" color="text.secondary">Variety</Typography>
                    <Chip
                      label={selectedPrediction.variety || 'Unknown'}
                      color="primary"
                      sx={{ mt: 0.5, fontWeight: 'bold' }}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2" color="text.secondary">Confidence</Typography>
                    <Typography variant="h6" sx={{ mt: 0.5 }}>
                      {selectedPrediction.confidence?.toFixed(2) || '0.00'}%
                    </Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="subtitle2" color="text.secondary">Date & Time</Typography>
                    <Typography variant="body2" sx={{ mt: 0.5 }}>
                      {formatDate(selectedPrediction.timestamp)}
                    </Typography>
                  </Grid>
                </Grid>

                {/* Top 3 Predictions */}
                {selectedPrediction.top_3_predictions && selectedPrediction.top_3_predictions.length > 0 && (
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Top 3 Predictions:
                    </Typography>
                    {selectedPrediction.top_3_predictions.map((pred, idx) => (
                      <Box
                        key={idx}
                        sx={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          p: 1.5,
                          mb: 1,
                          bgcolor: 'background.default',
                          borderRadius: 1
                        }}
                      >
                        <Typography>
                          {idx + 1}. {pred.variety}
                        </Typography>
                        <Chip
                          label={`${pred.confidence.toFixed(2)}%`}
                          size="small"
                          color={idx === 0 ? 'primary' : 'default'}
                        />
                      </Box>
                    ))}
                  </Box>
                )}

                {/* Features - Simplified Display */}
                {selectedPrediction.features && (
                  <Box>
                    <Typography variant="subtitle1" gutterBottom sx={{ mb: 2 }}>
                      <InfoIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                      Extracted Features
                    </Typography>
                    
                    {/* Name */}
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" color="primary" gutterBottom sx={{ fontWeight: 'bold' }}>
                        Name
                      </Typography>
                      <Chip
                        label={selectedPrediction.variety || 'Unknown'}
                        color="primary"
                        sx={{ fontWeight: 'bold' }}
                      />
                    </Box>
                    
                    {/* Color */}
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" color="primary" gutterBottom sx={{ fontWeight: 'bold' }}>
                        Color
                      </Typography>
                      <TableContainer component={Paper} variant="outlined">
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell><strong>Feature</strong></TableCell>
                              <TableCell align="right"><strong>Value</strong></TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {Object.entries(featureCategories['Color']).map(([displayName, featureKey]) => {
                              const value = selectedPrediction.features[featureKey];
                              if (value === undefined) return null;
                              return (
                                <TableRow key={featureKey}>
                                  <TableCell>{displayName}</TableCell>
                                  <TableCell align="right">
                                    {typeof value === 'number' ? value.toFixed(2) : value}
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Box>

                    {/* Texture */}
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" color="primary" gutterBottom sx={{ fontWeight: 'bold' }}>
                        Texture
                      </Typography>
                      <TableContainer component={Paper} variant="outlined">
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell><strong>Feature</strong></TableCell>
                              <TableCell align="right"><strong>Value</strong></TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {Object.entries(featureCategories['Texture']).map(([displayName, featureKey]) => {
                              const value = selectedPrediction.features[featureKey];
                              if (value === undefined) return null;
                              return (
                                <TableRow key={featureKey}>
                                  <TableCell>{displayName}</TableCell>
                                  <TableCell align="right">
                                    {typeof value === 'number' ? value.toFixed(4) : value}
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Box>

                    {/* Size (Area and Perimeter highlighted) */}
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" color="primary" gutterBottom sx={{ fontWeight: 'bold' }}>
                        Size
                      </Typography>
                      <TableContainer component={Paper} variant="outlined">
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell><strong>Feature</strong></TableCell>
                              <TableCell align="right"><strong>Value</strong></TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            <TableRow sx={{ bgcolor: 'action.hover' }}>
                              <TableCell><strong>Area</strong></TableCell>
                              <TableCell align="right">
                                <strong>{selectedPrediction.features['Size_Area']?.toFixed(2) || 'N/A'}</strong>
                              </TableCell>
                            </TableRow>
                            <TableRow sx={{ bgcolor: 'action.hover' }}>
                              <TableCell><strong>Perimeter</strong></TableCell>
                              <TableCell align="right">
                                <strong>{selectedPrediction.features['Size_Perimeter']?.toFixed(2) || 'N/A'}</strong>
                              </TableCell>
                            </TableRow>
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Box>

                    {/* Shape */}
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" color="primary" gutterBottom sx={{ fontWeight: 'bold' }}>
                        Shape
                      </Typography>
                      <TableContainer component={Paper} variant="outlined">
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell><strong>Feature</strong></TableCell>
                              <TableCell align="right"><strong>Value</strong></TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {Object.entries(featureCategories['Shape']).map(([displayName, featureKey]) => {
                              const value = selectedPrediction.features[featureKey];
                              if (value === undefined) return null;
                              return (
                                <TableRow key={featureKey}>
                                  <TableCell>{displayName}</TableCell>
                                  <TableCell align="right">
                                    {typeof value === 'number' ? value.toFixed(4) : value}
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Box>
                  </Box>
                )}

                {/* Error message if any */}
                {selectedPrediction.prediction_error && (
                  <Alert severity="warning" sx={{ mt: 2 }}>
                    {selectedPrediction.prediction_error}
                  </Alert>
                )}
              </Box>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseDialog}>Close</Button>
          </DialogActions>
        </Dialog>
      </motion.div>
    </Container>
  );
}

export default HistoryPage;
