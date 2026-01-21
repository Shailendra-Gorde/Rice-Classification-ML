import React, { useState, useRef } from 'react';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  Grid,
  Paper,
  Chip,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  CameraAlt as CameraIcon,
  PhotoCamera,
  CheckCircle,
  Refresh
} from '@mui/icons-material';
import { motion } from 'framer-motion';

const API_BASE_URL = 'http://localhost:5001/api';

function PredictPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [features, setFeatures] = useState(null);
  const [error, setError] = useState(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [stream, setStream] = useState(null);
  
  const fileInputRef = useRef(null);
  const cameraVideoRef = useRef(null);
  const canvasRef = useRef(null);

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please select an image file');
        return;
      }
      
      setSelectedImage(file);
      setError(null);
      setPrediction(null);
      setFeatures(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Handle camera capture
  const startCamera = async () => {
    try {
      // Check if getUserMedia is available
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError('Camera API not available in this browser. Please use a modern browser like Chrome, Firefox, or Safari.');
        return;
      }

      // Try different camera constraints for Mac
      let constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      };
      
      // On Mac, don't use facingMode (that's for mobile)
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      
      setStream(mediaStream);
      if (cameraVideoRef.current) {
        cameraVideoRef.current.srcObject = mediaStream;
        
        // Wait for video metadata and then play
        const video = cameraVideoRef.current;
        
        const handleLoadedMetadata = () => {
          video.play()
            .then(() => {
              console.log('Video playing successfully');
              setError(null);
            })
            .catch(err => {
              console.error('Error playing video:', err);
              setError('Camera started but video not playing. Try refreshing the page.');
            });
        };
        
        video.addEventListener('loadedmetadata', handleLoadedMetadata);
        video.addEventListener('loadeddata', () => {
          console.log('Video data loaded');
        });
        
        // Force play attempt
        setTimeout(() => {
          if (video.paused) {
            video.play().catch(err => console.error('Delayed play failed:', err));
          }
        }, 100);
      }
      setCameraOpen(true);
      setError(null);
    } catch (err) {
      let errorMessage = 'Could not access camera. ';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        errorMessage += 'Please allow camera access in your browser settings.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        errorMessage += 'No camera found on this device.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        errorMessage += 'Camera is being used by another application.';
      } else {
        errorMessage += `Error: ${err.message}`;
      }
      setError(errorMessage);
      console.error('Camera error:', err);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (cameraVideoRef.current) {
      cameraVideoRef.current.srcObject = null;
    }
    setCameraOpen(false);
  };

  const capturePhoto = () => {
    if (cameraVideoRef.current && canvasRef.current) {
      const video = cameraVideoRef.current;
      const canvas = canvasRef.current;
      
      // Check if video is ready
      if (video.readyState < 2) {
        setError('Camera not ready. Please wait a moment and try again.');
        return;
      }
      
      // Get actual video dimensions
      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;
      
      if (width === 0 || height === 0) {
        setError('Invalid video dimensions. Please try again.');
        return;
      }
      
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      
      // Draw video frame to canvas
      try {
        ctx.drawImage(video, 0, 0, width, height);
        
        // Convert canvas to blob
        canvas.toBlob((blob) => {
          if (!blob) {
            setError('Failed to capture image. Please try again.');
            return;
          }
          
          const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
          setSelectedImage(file);
          
          // Create preview from canvas
          const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
          setPreview(dataUrl);
          
          // Close camera
          stopCamera();
          
          // Clear any errors
          setError(null);
        }, 'image/jpeg', 0.95);
      } catch (err) {
        console.error('Error capturing photo:', err);
        setError('Failed to capture photo. Please try again.');
      }
    } else {
      setError('Camera not initialized. Please try opening the camera again.');
    }
  };

  // Handle prediction
  const handlePredict = async () => {
    if (!selectedImage) {
      setError('Please select or capture an image first');
      return;
    }

    setPredicting(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedImage);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setPrediction(data);
        setFeatures(data.extracted_features);
        
        // Show warning if prediction failed but features were extracted
        if (data.prediction_error) {
          setError(`⚠️ ${data.prediction_error} Features extracted successfully.`);
        }
        
        // Trigger dashboard data refresh after successful prediction
        if (data.success && data.prediction) {
          // Notify parent to refresh dashboard data
          window.dispatchEvent(new CustomEvent('predictionMade', { detail: data }));
        }
      }
    } catch (err) {
      setError(`Error: ${err.message}. Make sure the API server is running on port 5000.`);
      console.error('Prediction error:', err);
    } finally {
      setPredicting(false);
    }
  };

  // Reset all
  const handleReset = () => {
    setSelectedImage(null);
    setPreview(null);
    setPrediction(null);
    setFeatures(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Feature categories as requested by user
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

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold', mb: 4 }}>
          🍚 Rice Variety Classification
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
          Upload an image or capture a photo to identify Indian rice varieties
        </Typography>

        {/* Error Alert */}
        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        <Grid container spacing={3}>
          {/* Left Column - Image Upload and Controls */}
          <Grid item xs={12} md={5}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Upload Image
                </Typography>

                {/* Image Preview */}
                <Box
                  sx={{
                    width: '100%',
                    minHeight: '300px',
                    border: '2px dashed',
                    borderColor: 'divider',
                    borderRadius: 2,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    p: 2,
                    mb: 2,
                    bgcolor: 'background.default',
                    position: 'relative',
                    overflow: 'hidden'
                  }}
                >
                  {preview ? (
                    <Box
                      component="img"
                      src={preview}
                      alt="Preview"
                      sx={{
                        maxWidth: '100%',
                        maxHeight: '400px',
                        objectFit: 'contain',
                        borderRadius: 1
                      }}
                    />
                  ) : (
                    <Box sx={{ textAlign: 'center' }}>
                      <PhotoCamera sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                      <Typography variant="body2" color="text.secondary">
                        No image selected
                      </Typography>
                    </Box>
                  )}
                </Box>

                {/* Upload Buttons */}
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Button
                      fullWidth
                      variant="contained"
                      component="label"
                      startIcon={<CloudUploadIcon />}
                      sx={{ mb: 2 }}
                    >
                      Upload Image
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        hidden
                        onChange={handleFileChange}
                      />
                    </Button>
                  </Grid>
                  <Grid item xs={6}>
                    <Button
                      fullWidth
                      variant="outlined"
                      startIcon={<CameraIcon />}
                      onClick={startCamera}
                      sx={{ mb: 2 }}
                    >
                      Use Camera
                    </Button>
                  </Grid>
                </Grid>

                {/* Action Buttons */}
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Button
                      fullWidth
                      variant="contained"
                      color="primary"
                      onClick={handlePredict}
                      disabled={!selectedImage || predicting}
                      startIcon={predicting ? <CircularProgress size={20} /> : <CheckCircle />}
                    >
                      {predicting ? 'Analyzing...' : 'Predict Variety'}
                    </Button>
                  </Grid>
                  <Grid item xs={6}>
                    <Button
                      fullWidth
                      variant="outlined"
                      onClick={handleReset}
                      startIcon={<Refresh />}
                    >
                      Reset
                    </Button>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Right Column - Results */}
          <Grid item xs={12} md={7}>
            {/* Prediction Results */}
            {prediction && (
              <Card sx={{ mb: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Prediction Results
                  </Typography>
                  
                  {prediction.prediction && prediction.prediction !== "Model needs retraining" && !prediction.prediction_error ? (
                    <>
                      <Box sx={{ my: 3, textAlign: 'center' }}>
                        <Chip
                          label={prediction.prediction}
                          color="primary"
                          sx={{
                            fontSize: '1.2rem',
                            py: 3,
                            px: 2,
                            fontWeight: 'bold'
                          }}
                        />
                        <Typography variant="h4" sx={{ mt: 2, fontWeight: 'bold' }}>
                          {prediction.confidence.toFixed(2)}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Confidence Score
                        </Typography>
                      </Box>
                    </>
                  ) : (
                    <Box sx={{ my: 3, textAlign: 'center' }}>
                      <Alert severity="warning" sx={{ mb: 2 }}>
                        Model needs to be retrained with image-extracted features (24 features) to make accurate predictions.
                      </Alert>
                      <Typography variant="body1" color="text.secondary">
                        Features have been extracted successfully. Please retrain the model to see predictions.
                      </Typography>
                    </Box>
                  )}

                  {/* Top 3 Predictions */}
                  {prediction.top_3_predictions && prediction.top_3_predictions.length > 0 && (
                    <Box sx={{ mt: 3 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        Top 3 Predictions:
                      </Typography>
                      {prediction.top_3_predictions.map((pred, idx) => (
                        <Box
                          key={idx}
                          sx={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            p: 1.5,
                            mb: 1,
                            bgcolor: 'background.default',
                            borderRadius: 1,
                            border: idx === 0 ? '2px solid' : '1px solid',
                            borderColor: idx === 0 ? 'primary.main' : 'divider'
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
                </CardContent>
              </Card>
            )}

            {/* Extracted Features - Simplified Display */}
            {features && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
                    Extracted Features
                  </Typography>
                  
                  {/* Name (Rice Variety) */}
                  {prediction && prediction.prediction && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle2" color="primary" gutterBottom>
                        Name
                      </Typography>
                      <Chip
                        label={prediction.prediction}
                        color="primary"
                        sx={{ fontSize: '1rem', py: 1, px: 2, fontWeight: 'bold' }}
                      />
                    </Box>
                  )}
                  
                  {/* Color Features */}
                  <Box sx={{ mb: 3 }}>
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
                            const value = features[featureKey];
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

                  {/* Texture Features */}
                  <Box sx={{ mb: 3 }}>
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
                            const value = features[featureKey];
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

                  {/* Size Features (Area and Perimeter highlighted) */}
                  <Box sx={{ mb: 3 }}>
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
                              <strong>{features['Size_Area']?.toFixed(2) || 'N/A'}</strong>
                            </TableCell>
                          </TableRow>
                          <TableRow sx={{ bgcolor: 'action.hover' }}>
                            <TableCell><strong>Perimeter</strong></TableCell>
                            <TableCell align="right">
                              <strong>{features['Size_Perimeter']?.toFixed(2) || 'N/A'}</strong>
                            </TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Box>

                  {/* Shape Features */}
                  <Box sx={{ mb: 3 }}>
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
                            const value = features[featureKey];
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
                </CardContent>
              </Card>
            )}

            {/* Empty State */}
            {!prediction && (
              <Card>
                <CardContent>
                  <Typography variant="body1" color="text.secondary" align="center" sx={{ py: 4 }}>
                    Upload an image or capture a photo to see prediction results and extracted features
                  </Typography>
                </CardContent>
              </Card>
            )}
          </Grid>
        </Grid>
      </motion.div>

      {/* Camera Dialog */}
      <Dialog
        open={cameraOpen}
        onClose={stopCamera}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            backgroundColor: 'background.paper'
          }
        }}
      >
        <DialogTitle>Capture Photo</DialogTitle>
        <DialogContent>
          <Box sx={{ position: 'relative', width: '100%', minHeight: '400px', backgroundColor: '#000', borderRadius: 1, overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Box
              component="video"
              ref={cameraVideoRef}
              autoPlay
              playsInline
              muted
              style={{
                width: '100%',
                maxWidth: '100%',
                height: 'auto',
                maxHeight: '500px',
                objectFit: 'contain',
                display: 'block',
                backgroundColor: '#000'
              }}
            />
            <canvas 
              ref={canvasRef} 
              style={{ display: 'none' }}
            />
            {!stream && (
              <Box
                sx={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  color: 'white',
                  textAlign: 'center',
                  zIndex: 10
                }}
              >
                <CircularProgress sx={{ color: 'white', mb: 2 }} />
                <Typography color="white">Starting camera...</Typography>
              </Box>
            )}
            {stream && cameraVideoRef.current && cameraVideoRef.current.readyState < 2 && (
              <Box
                sx={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  color: 'white',
                  textAlign: 'center',
                  zIndex: 10
                }}
              >
                <CircularProgress sx={{ color: 'white', mb: 2 }} />
                <Typography color="white">Loading video stream...</Typography>
              </Box>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={stopCamera}>Cancel</Button>
          <Button 
            onClick={capturePhoto} 
            variant="contained" 
            startIcon={<PhotoCamera />}
            disabled={!stream}
          >
            Capture
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}

export default PredictPage;
