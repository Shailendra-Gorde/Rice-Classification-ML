import React, { useState, useEffect } from 'react';
import {
  Box,
  Tabs,
  Tab,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import {
  Analytics as AnalyticsIcon,
  EmojiEvents as TrophyIcon,
  CheckCircle as CheckIcon
} from '@mui/icons-material';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell
} from 'recharts';
import { motion } from 'framer-motion';

const AnalysisPage = ({ data, onDataRefresh }) => {
  const [selectedTab, setSelectedTab] = useState(0);
  const [refreshKey, setRefreshKey] = useState(0);

  // Auto-refresh when new predictions are made
  useEffect(() => {
    // Listen for prediction events
    const handlePrediction = () => {
      console.log('AnalysisPage: Prediction made, refreshing data...');
      if (onDataRefresh) {
        onDataRefresh();
      }
      setRefreshKey(prev => prev + 1);
    };
    
    window.addEventListener('predictionMade', handlePrediction);
    
    // Auto-refresh every 5 seconds
    const interval = setInterval(() => {
      if (onDataRefresh) {
        onDataRefresh();
      }
    }, 5000);
    
    return () => {
      window.removeEventListener('predictionMade', handlePrediction);
      clearInterval(interval);
    };
  }, [onDataRefresh]);

  if (!data) return <Typography>Loading...</Typography>;

  const { all_models, best_model, model_rankings, prediction_history } = data || {};

  // Ensure model_rankings is an array
  const rankings = model_rankings && Array.isArray(model_rankings) ? model_rankings : [];

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };

  const currentModel = all_models && all_models[selectedTab] ? all_models[selectedTab] : {
    'name': 'Random Forest',
    'Accuracy': 0.7365,
    'Precision': 0.70,
    'Recall': 0.70,
    'F1-Score': 0.70,
    'ROC-AUC': 0.72
  };

  const metricsData = [
    { metric: 'Accuracy', value: (currentModel.Accuracy * 100).toFixed(2) },
    { metric: 'Precision', value: (currentModel.Precision * 100).toFixed(2) },
    { metric: 'Recall', value: (currentModel.Recall * 100).toFixed(2) },
    { metric: 'F1-Score', value: (currentModel['F1-Score'] * 100).toFixed(2) },
    { metric: 'ROC-AUC', value: (currentModel['ROC-AUC'] * 100).toFixed(2) }
  ];

  const radarData = [
    { metric: 'Accuracy', value: currentModel.Accuracy * 100, fullMark: 100 },
    { metric: 'Precision', value: currentModel.Precision * 100, fullMark: 100 },
    { metric: 'Recall', value: currentModel.Recall * 100, fullMark: 100 },
    { metric: 'F1-Score', value: currentModel['F1-Score'] * 100, fullMark: 100 },
    { metric: 'ROC-AUC', value: currentModel['ROC-AUC'] * 100, fullMark: 100 }
  ];

  const confusionMatrix = currentModel?.confusion_matrix || best_model?.confusion_matrix;
  
  // For multi-class (13 varieties), confusion matrix is complex
  // We'll show a simplified view or note
  const isMultiClass = data?.project?.num_classes > 2;

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#8dd1e1'];

  return (
    <Box>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
            <AnalyticsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Model Analysis
          </Typography>
          {prediction_history?.total > 0 && (
            <Chip 
              label={`${prediction_history.total} predictions made`} 
              color="primary"
              sx={{ fontWeight: 'bold' }}
            />
          )}
        </Box>
      </motion.div>

      {/* Best Model Banner */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <TrophyIcon sx={{ fontSize: 40, color: '#ffd700', mr: 2 }} />
                <Box>
                  <Typography variant="h5" sx={{ fontWeight: 'bold', color: 'white' }}>
                    Best Model: {best_model?.name || 'Random Forest'}
                  </Typography>
                  <Typography variant="body1" sx={{ color: 'rgba(255,255,255,0.9)' }}>
                    Accuracy: {best_model?.performance ? (best_model.performance.accuracy * 100).toFixed(2) : '73.65'}% |
                    F1-Score: {best_model?.performance ? best_model.performance.f1_score.toFixed(4) : '0.7000'} |
                    ROC-AUC: {best_model?.performance ? best_model.performance.roc_auc.toFixed(4) : '0.7200'} |
                    Predictions: {best_model?.predictions_made || prediction_history?.total || 0}
                  </Typography>
                </Box>
              </Box>
              <Chip
                icon={<CheckIcon />}
                label="RECOMMENDED"
                sx={{ backgroundColor: '#ffd700', color: '#000', fontWeight: 'bold' }}
              />
            </Box>
          </CardContent>
        </Card>
      </motion.div>

      {/* Model Rankings with Prediction Statistics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', mb: 2 }}>
              Model Performance & Predictions
            </Typography>
            
            {rankings && rankings.length > 0 ? (
              <>
                <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell><strong>Rank</strong></TableCell>
                        <TableCell><strong>Model</strong></TableCell>
                        <TableCell align="right"><strong>F1-Score</strong></TableCell>
                        <TableCell align="right"><strong>Model Accuracy</strong></TableCell>
                        <TableCell align="right"><strong>Avg Confidence</strong></TableCell>
                        <TableCell align="right"><strong>Predictions</strong></TableCell>
                        <TableCell align="right"><strong>Varieties Found</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {rankings.map((model, index) => (
                        <TableRow 
                          key={index}
                          sx={{ 
                            bgcolor: index === 0 ? 'action.selected' : 'transparent',
                            '&:hover': { bgcolor: 'action.hover' }
                          }}
                        >
                          <TableCell>
                            <Chip 
                              label={`#${index + 1}`}
                              color={index === 0 ? 'primary' : 'default'}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              {index === 0 && <TrophyIcon sx={{ color: '#ffd700', fontSize: 20 }} />}
                              <Typography variant="body1" sx={{ fontWeight: index === 0 ? 'bold' : 'normal' }}>
                                {model.model}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body1" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                              {model.score ? model.score.toFixed(4) : 'N/A'}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                              {model.accuracy ? `${(model.accuracy * 100).toFixed(2)}%` : 'N/A'}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Chip 
                              label={model.avg_confidence ? `${model.avg_confidence.toFixed(1)}%` : 'N/A'}
                              color={model.avg_confidence && model.avg_confidence > 70 ? 'success' : model.avg_confidence && model.avg_confidence > 50 ? 'warning' : 'default'}
                              size="small"
                            />
                          </TableCell>
                          <TableCell align="right">
                            <Chip 
                              label={model.predictions || 0}
                              color="secondary"
                              size="small"
                            />
                          </TableCell>
                          <TableCell align="right">
                            <Chip 
                              label={model.varieties_predicted || 0}
                              color="info"
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
                
                {/* Prediction Statistics */}
                {prediction_history?.statistics && (
                  <Box sx={{ mt: 3, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
                    <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
                      Prediction Statistics
                    </Typography>
                    <Grid container spacing={2} sx={{ mt: 1 }}>
                      <Grid item xs={6} sm={3}>
                        <Typography variant="caption" color="text.secondary">High Confidence</Typography>
                        <Typography variant="h6" color="success.main">
                          {prediction_history.statistics.high_confidence_predictions || 0}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">(&gt;80%)</Typography>
                      </Grid>
                      <Grid item xs={6} sm={3}>
                        <Typography variant="caption" color="text.secondary">Medium Confidence</Typography>
                        <Typography variant="h6" color="warning.main">
                          {prediction_history.statistics.medium_confidence_predictions || 0}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">(50-80%)</Typography>
                      </Grid>
                      <Grid item xs={6} sm={3}>
                        <Typography variant="caption" color="text.secondary">Low Confidence</Typography>
                        <Typography variant="h6" color="error.main">
                          {prediction_history.statistics.low_confidence_predictions || 0}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">(&lt;50%)</Typography>
                      </Grid>
                      <Grid item xs={6} sm={3}>
                        <Typography variant="caption" color="text.secondary">Avg Confidence</Typography>
                        <Typography variant="h6">
                          {prediction_history.statistics.avg_confidence ? `${prediction_history.statistics.avg_confidence.toFixed(1)}%` : 'N/A'}
                        </Typography>
                      </Grid>
                    </Grid>
                  </Box>
                )}
                
                {/* Variety Distribution */}
                {prediction_history?.variety_distribution && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
                      Predicted Varieties Distribution
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                      {Object.entries(prediction_history.variety_distribution)
                        .filter(([_, count]) => count > 0)
                        .sort(([_, a], [__, b]) => b - a)
                        .map(([variety, count]) => (
                          <Chip
                            key={variety}
                            label={`${variety}: ${count}`}
                            color="primary"
                            variant="outlined"
                            size="small"
                          />
                        ))}
                    </Box>
                  </Box>
                )}
              </>
            ) : (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="body1" color="text.secondary">
                  No model data available. Make some predictions to see model performance.
                </Typography>
              </Box>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Model Tabs */}
      <Card sx={{ mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={selectedTab}
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
            sx={{
              '& .MuiTab-root': {
                minWidth: 120,
                fontWeight: 'bold'
              }
            }}
          >
            {all_models.map((model, index) => (
              <Tab
                key={model.Model}
                label={model.Model}
                icon={index === 0 ? <TrophyIcon fontSize="small" sx={{ color: '#ffd700' }} /> : null}
                iconPosition="start"
              />
            ))}
          </Tabs>
        </Box>

        <CardContent>
          <motion.div
            key={selectedTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4 }}
          >
            {/* Model Header */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="h5" sx={{ fontWeight: 'bold', mb: 1 }}>
                {currentModel.Model}
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                <Chip
                  label={`Rank #${selectedTab + 1}`}
                  color={selectedTab === 0 ? 'warning' : 'default'}
                  size="small"
                />
                <Chip
                  label={`Accuracy: ${(currentModel.Accuracy * 100).toFixed(2)}%`}
                  color="primary"
                  size="small"
                />
                <Chip
                  label={`F1: ${currentModel['F1-Score'].toFixed(4)}`}
                  color="secondary"
                  size="small"
                />
              </Box>
            </Box>

            {/* Metrics Cards */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} md={2.4}>
                <Card sx={{ textAlign: 'center', backgroundColor: '#1a2744' }}>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom variant="body2">
                      Accuracy
                    </Typography>
                    <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#90caf9' }}>
                      {(currentModel.Accuracy * 100).toFixed(2)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={2.4}>
                <Card sx={{ textAlign: 'center', backgroundColor: '#1a2744' }}>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom variant="body2">
                      Precision
                    </Typography>
                    <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#82ca9d' }}>
                      {(currentModel.Precision * 100).toFixed(2)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={2.4}>
                <Card sx={{ textAlign: 'center', backgroundColor: '#1a2744' }}>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom variant="body2">
                      Recall
                    </Typography>
                    <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#ffc658' }}>
                      {(currentModel.Recall * 100).toFixed(2)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={2.4}>
                <Card sx={{ textAlign: 'center', backgroundColor: '#1a2744' }}>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom variant="body2">
                      F1-Score
                    </Typography>
                    <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#ff8042' }}>
                      {currentModel['F1-Score'].toFixed(4)}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={2.4}>
                <Card sx={{ textAlign: 'center', backgroundColor: '#1a2744' }}>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom variant="body2">
                      ROC-AUC
                    </Typography>
                    <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#8dd1e1' }}>
                      {currentModel['ROC-AUC'].toFixed(4)}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Charts Row */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
                      Performance Radar
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <RadarChart data={radarData}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="metric" />
                        <PolarRadiusAxis angle={90} domain={[0, 100]} />
                        <Radar
                          name={currentModel.Model}
                          dataKey="value"
                          stroke="#8884d8"
                          fill="#8884d8"
                          fillOpacity={0.6}
                        />
                        <Tooltip />
                      </RadarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
                      Metrics Breakdown
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={metricsData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="metric"
                          angle={-45}
                          textAnchor="end"
                          height={80}
                          interval={0}
                        />
                        <YAxis domain={[0, 100]} />
                        <Tooltip />
                        <Bar dataKey="value" name="Score (%)">
                          {metricsData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Confusion Matrix - Multi-class or Binary */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', mb: 2 }}>
                  Model Performance Details
                </Typography>
                {isMultiClass ? (
                  <Box>
                    <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
                      This is a multi-class classification model for {data?.project?.num_classes || 13} Indian rice varieties.
                      Confusion matrix visualization for {data?.project?.num_classes || 13} classes would be too large to display here.
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                      {data?.project?.class_names?.slice(0, 13).map((variety, idx) => (
                        <Chip 
                          key={idx} 
                          label={variety} 
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                      ))}
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      Metrics shown above (Accuracy, Precision, Recall, F1-Score, ROC-AUC) are macro-averaged across all {data?.project?.num_classes || 13} classes.
                    </Typography>
                  </Box>
                ) : (
                  <Box>
                    <TableContainer component={Paper} sx={{ maxWidth: 600, mx: 'auto' }}>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#1a2744' }}>
                              Actual \ Predicted
                            </TableCell>
                            <TableCell align="center" sx={{ fontWeight: 'bold', backgroundColor: '#1a2744' }}>
                              Predicted Negative
                            </TableCell>
                            <TableCell align="center" sx={{ fontWeight: 'bold', backgroundColor: '#1a2744' }}>
                              Predicted Positive
                            </TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          <TableRow>
                            <TableCell sx={{ fontWeight: 'bold' }}>Actual Negative</TableCell>
                            <TableCell align="center" sx={{ backgroundColor: '#2e7d32', color: 'white', fontSize: '1.2rem' }}>
                              {confusionMatrix?.true_negatives || 0}
                            </TableCell>
                            <TableCell align="center" sx={{ backgroundColor: '#d32f2f', color: 'white', fontSize: '1.2rem' }}>
                              {confusionMatrix?.false_positives || 0}
                            </TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell sx={{ fontWeight: 'bold' }}>Actual Positive</TableCell>
                            <TableCell align="center" sx={{ backgroundColor: '#d32f2f', color: 'white', fontSize: '1.2rem' }}>
                              {confusionMatrix?.false_negatives || 0}
                            </TableCell>
                            <TableCell align="center" sx={{ backgroundColor: '#2e7d32', color: 'white', fontSize: '1.2rem' }}>
                              {confusionMatrix?.true_positives || 0}
                            </TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </TableContainer>
                    <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, flexWrap: 'wrap', mt: 2 }}>
                      <Chip label={`TN: ${confusionMatrix?.true_negatives || 0}`} sx={{ backgroundColor: '#2e7d32' }} />
                      <Chip label={`FP: ${confusionMatrix?.false_positives || 0}`} sx={{ backgroundColor: '#d32f2f' }} />
                      <Chip label={`FN: ${confusionMatrix?.false_negatives || 0}`} sx={{ backgroundColor: '#d32f2f' }} />
                      <Chip label={`TP: ${confusionMatrix?.true_positives || 0}`} sx={{ backgroundColor: '#2e7d32' }} />
                    </Box>
                  </Box>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </CardContent>
      </Card>
    </Box>
  );
};

export default AnalysisPage;
