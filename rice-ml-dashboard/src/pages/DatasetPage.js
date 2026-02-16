import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress
} from '@mui/material';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { motion } from 'framer-motion';
import {
  DataUsage as DataIcon,
  TrendingUp as TrendingIcon,
  Calculate as CalculateIcon,
  Info as InfoIcon
} from '@mui/icons-material';

const DatasetPage = ({ data }) => {
  if (!data) return <Typography>Loading...</Typography>;

  const { project, dataset, feature_importance, prediction_history } = data;

  // Use dataset class distribution (from actual dataset images, not predictions)
  const classDistribution = dataset.class_distribution || {};
  // Show all classes from the dataset, even if count is 0
  const classDistData = project.class_names
    ? project.class_names.map(variety => ({
        name: variety,
        value: classDistribution[variety] || 0
      }))
    : Object.entries(classDistribution)
        .map(([name, value]) => ({
          name,
          value: value || 0
        }))
        .sort((a, b) => b.value - a.value); // Sort by count descending

  const featureImportanceData = feature_importance.top_5;

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#8dd1e1'];

  const statsData = Object.entries(dataset.statistics).slice(0, 5);

  return (
    <Box>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold', mb: 3 }}>
          <DataIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Dataset Overview
        </Typography>
      </motion.div>

      {/* Project Info Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <motion.div
            whileHover={{ scale: 1.02 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Total Samples
                </Typography>
                <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#90caf9' }}>
                  {project.total_samples || 0}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Rice grain samples in dataset
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} md={3}>
          <motion.div
            whileHover={{ scale: 1.02 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Features
                </Typography>
                <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#82ca9d' }}>
                  {project.num_features}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Image-extracted features
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} md={3}>
          <motion.div
            whileHover={{ scale: 1.02 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Classes
                </Typography>
                <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#ffc658' }}>
                  {project.num_classes}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Indian rice varieties
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} md={3}>
          <motion.div
            whileHover={{ scale: 1.02 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Imbalance Ratio
                </Typography>
                <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#ff8042' }}>
                  {dataset.imbalance_ratio ? dataset.imbalance_ratio.toFixed(2) : '1.00'}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Dataset distribution ratio
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Charts Row */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', mb: 2 }}>
                  Class Distribution
                  <Chip 
                    label="Dataset images" 
                    size="small" 
                    sx={{ ml: 2 }}
                    color="primary"
                  />
                </Typography>
                {classDistData.length > 0 ? (
                  <>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={classDistData.filter(d => d.value > 0)}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent, value }) => {
                            return `${name}: ${value} (${(percent * 100).toFixed(0)}%)`;
                          }}
                          outerRadius={100}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {classDistData.filter(d => d.value > 0).map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                    <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 1, flexWrap: 'wrap' }}>
                      {classDistData.map((entry, index) => (
                        <Chip
                          key={entry.name}
                          label={`${entry.name}: ${entry.value}`}
                          sx={{ backgroundColor: entry.value > 0 ? COLORS[index % COLORS.length] : '#e0e0e0' }}
                          size="small"
                        />
                      ))}
                    </Box>
                  </>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Typography variant="body2" color="text.secondary">
                      No dataset images found. Add images to data/images/ folders.
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', mb: 2 }}>
                  <TrendingIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Top 5 Important Features
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={featureImportanceData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="feature" type="category" width={120} />
                    <Tooltip />
                    <Bar dataKey="importance" fill="#90caf9">
                      {featureImportanceData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Features Info */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', mb: 2 }}>
                  <InfoIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  All Features
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {dataset.features.all_features?.map((feature) => (
                    <Chip
                      key={feature}
                      label={feature.replace(/_/g, ' ')}
                      color="primary"
                      variant="outlined"
                      size="small"
                    />
                  )) || <Typography>Loading features...</Typography>}
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', mb: 2 }}>
                  <CalculateIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Texture Features (6)
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {dataset.features.texture_features?.map((feature) => (
                    <Chip
                      key={feature}
                      label={feature.replace(/_/g, ' ')}
                      color="secondary"
                      variant="filled"
                      size="small"
                    />
                  )) || <Typography>Loading features...</Typography>}
                </Box>
                <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                  GLCM and LBP texture measurements
                </Typography>
                <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
                  Features are automatically extracted from uploaded rice grain images
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Feature Descriptions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.6 }}
      >
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', mb: 2 }}>
              Feature Descriptions
            </Typography>
            <TableContainer component={Paper} sx={{ backgroundColor: 'transparent' }}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold' }}>Feature</TableCell>
                    <TableCell sx={{ fontWeight: 'bold' }}>Description</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(dataset.features.feature_descriptions).map(([feature, description]) => (
                    <TableRow key={feature} hover>
                      <TableCell>
                        <Chip label={feature} size="small" />
                      </TableCell>
                      <TableCell>{description}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      </motion.div>

      {/* Feature Statistics Sample */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.7 }}
      >
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', mb: 2 }}>
              Feature Statistics
              {prediction_history?.total && (
                <Chip 
                  label="Live from predictions" 
                  size="small" 
                  sx={{ ml: 2 }}
                  color="success"
                />
              )}
            </Typography>
            {Object.keys(dataset.statistics || {}).length > 0 ? (
              Object.entries(dataset.statistics).slice(0, 10).map(([feature, stats]) => (
              <Box key={feature} sx={{ mb: 3 }}>
                <Typography variant="body1" sx={{ fontWeight: 'bold', mb: 1 }}>
                  {feature}
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6} md={2}>
                    <Typography variant="caption" color="textSecondary">Mean</Typography>
                    <Typography variant="body2">{stats.mean.toFixed(2)}</Typography>
                  </Grid>
                  <Grid item xs={6} md={2}>
                    <Typography variant="caption" color="textSecondary">Std Dev</Typography>
                    <Typography variant="body2">{stats.std.toFixed(2)}</Typography>
                  </Grid>
                  <Grid item xs={6} md={2}>
                    <Typography variant="caption" color="textSecondary">Min</Typography>
                    <Typography variant="body2">{stats.min.toFixed(2)}</Typography>
                  </Grid>
                  <Grid item xs={6} md={2}>
                    <Typography variant="caption" color="textSecondary">Median</Typography>
                    <Typography variant="body2">{stats.median.toFixed(2)}</Typography>
                  </Grid>
                  <Grid item xs={6} md={2}>
                    <Typography variant="caption" color="textSecondary">Max</Typography>
                    <Typography variant="body2">{stats.max.toFixed(2)}</Typography>
                  </Grid>
                </Grid>
                <Box sx={{ mt: 1 }}>
                  <LinearProgress
                    variant="determinate"
                    value={((stats.mean - stats.min) / (stats.max - stats.min)) * 100}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                </Box>
              </Box>
              ))
            ) : (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="body2" color="text.secondary">
                  Feature statistics will appear after making predictions.
                </Typography>
              </Box>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </Box>
  );
};

export default DatasetPage;
