import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  CardMedia,
  Grid,
  Button,
  Paper,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip
} from '@mui/material';
import {
  CompareArrows as CompareIcon,
  CloudUpload as UploadIcon,
  Image as ImageIcon
} from '@mui/icons-material';
import { motion } from 'framer-motion';

const API_BASE_URL = 'http://localhost:5001/api';

function ComparePage() {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [preview1, setPreview1] = useState(null);
  const [preview2, setPreview2] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleFile1 = (e) => {
    const f = e.target.files?.[0];
    if (f && f.type.startsWith('image/')) {
      setFile1(f);
      setError(null);
      setResult(null);
      const reader = new FileReader();
      reader.onloadend = () => setPreview1(reader.result);
      reader.readAsDataURL(f);
    }
  };

  const handleFile2 = (e) => {
    const f = e.target.files?.[0];
    if (f && f.type.startsWith('image/')) {
      setFile2(f);
      setError(null);
      setResult(null);
      const reader = new FileReader();
      reader.onloadend = () => setPreview2(reader.result);
      reader.readAsDataURL(f);
    }
  };

  const handleCompare = async () => {
    if (!file1 || !file2) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('image1', file1);
      formData.append('image2', file2);
      const response = await fetch(`${API_BASE_URL}/compare`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
        setResult(null);
      } else {
        setResult(data);
      }
    } catch (err) {
      setError(err.message || 'Compare failed. Ensure API is running on port 5001.');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile1(null);
    setFile2(null);
    setPreview1(null);
    setPreview2(null);
    setResult(null);
    setError(null);
  };

  const keyFeatures = ['Color_Mean_R', 'Color_Mean_G', 'Color_Mean_B', 'Shape_AspectRatio', 'Shape_Roundness', 'Size_Area', 'Texture_Contrast'];

  return (
    <Container maxWidth="xl">
      <Typography variant="h5" sx={{ mb: 2, fontWeight: 600 }}>
        Compare two rice samples
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Upload two images to see side-by-side variety prediction and key features. Useful for traders comparing samples.
      </Typography>

      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={5}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>Sample 1</Typography>
            <Box sx={{ mb: 2, minHeight: 200, bgcolor: 'action.hover', borderRadius: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
              {preview1 ? (
                <img src={preview1} alt="Sample 1" style={{ maxWidth: '100%', maxHeight: 280, objectFit: 'contain' }} />
              ) : (
                <Box sx={{ textAlign: 'center', color: 'text.secondary' }}>
                  <UploadIcon sx={{ fontSize: 48 }} />
                  <Typography variant="body2">Choose image</Typography>
                </Box>
              )}
            </Box>
            <Button variant="outlined" component="label" fullWidth startIcon={<ImageIcon />}>
              {file1 ? file1.name : 'Select image 1'}
              <input type="file" accept="image/*" hidden onChange={handleFile1} />
            </Button>
          </Paper>
        </Grid>
        <Grid item xs={12} md={2} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <CompareIcon sx={{ fontSize: 40, color: 'primary.main' }} />
        </Grid>
        <Grid item xs={12} md={5}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>Sample 2</Typography>
            <Box sx={{ mb: 2, minHeight: 200, bgcolor: 'action.hover', borderRadius: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
              {preview2 ? (
                <img src={preview2} alt="Sample 2" style={{ maxWidth: '100%', maxHeight: 280, objectFit: 'contain' }} />
              ) : (
                <Box sx={{ textAlign: 'center', color: 'text.secondary' }}>
                  <UploadIcon sx={{ fontSize: 48 }} />
                  <Typography variant="body2">Choose image</Typography>
                </Box>
              )}
            </Box>
            <Button variant="outlined" component="label" fullWidth startIcon={<ImageIcon />}>
              {file2 ? file2.name : 'Select image 2'}
              <input type="file" accept="image/*" hidden onChange={handleFile2} />
            </Button>
          </Paper>
        </Grid>
      </Grid>

      <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
        <Button
          variant="contained"
          onClick={handleCompare}
          disabled={!file1 || !file2 || loading}
          startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <CompareIcon />}
        >
          {loading ? 'Comparing…' : 'Compare'}
        </Button>
        <Button variant="outlined" onClick={reset}>Reset</Button>
      </Box>

      {result && result.success && (
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
          <Typography variant="h6" sx={{ mt: 4, mb: 2 }}>Comparison results</Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" color="primary" gutterBottom>Sample 1</Typography>
                  <Typography variant="h6">{result.result1.prediction}</Typography>
                  <Chip label={`${Number(result.result1.confidence).toFixed(1)}% confidence`} size="small" sx={{ mt: 1, mb: 1 }} />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>Top 3:</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                    {(result.result1.top_3_predictions || []).map((p, i) => (
                      <Chip key={i} label={`${p.variety} (${Number(p.confidence).toFixed(0)}%)`} size="small" variant="outlined" />
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" color="primary" gutterBottom>Sample 2</Typography>
                  <Typography variant="h6">{result.result2.prediction}</Typography>
                  <Chip label={`${Number(result.result2.confidence).toFixed(1)}% confidence`} size="small" sx={{ mt: 1, mb: 1 }} />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>Top 3:</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                    {(result.result2.top_3_predictions || []).map((p, i) => (
                      <Chip key={i} label={`${p.variety} (${Number(p.confidence).toFixed(0)}%)`} size="small" variant="outlined" />
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Typography variant="subtitle2" sx={{ mt: 3, mb: 1 }}>Key features comparison</Typography>
          <TableContainer component={Paper} sx={{ maxHeight: 320 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Feature</strong></TableCell>
                  <TableCell align="right"><strong>Sample 1</strong></TableCell>
                  <TableCell align="right"><strong>Sample 2</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {keyFeatures.filter(f => result.result1.extracted_features && f in result.result1.extracted_features).map(f => (
                  <TableRow key={f}>
                    <TableCell>{f.replace(/_/g, ' ')}</TableCell>
                    <TableCell align="right">{Number(result.result1.extracted_features[f]).toFixed(3)}</TableCell>
                    <TableCell align="right">{Number(result.result2.extracted_features[f]).toFixed(3)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </motion.div>
      )}
    </Container>
  );
}

export default ComparePage;
