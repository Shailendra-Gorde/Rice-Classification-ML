import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
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
  TableRow
} from '@mui/material';
import {
  AdminPanelSettings as AdminIcon,
  Download as DownloadIcon,
  BarChart as ChartIcon,
  ShoppingCart as CartIcon,
  List as ListIcon,
  TrendingUp as TrendingIcon
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import { motion } from 'framer-motion';

const API_BASE_URL = 'http://localhost:5001/api';

function AdminPage() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/admin/stats`);
      const data = await response.json();
      if (data.success) {
        setStats(data.stats);
      } else {
        setError(data.error || 'Failed to load stats');
      }
    } catch (err) {
      setError(err.message || 'Ensure API is running on port 5001.');
      setStats(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStats();
  }, []);

  const exportPurchases = () => {
    window.open(`${API_BASE_URL.replace('/api', '')}/api/admin/export/purchases`, '_blank');
  };

  const exportListingsCSV = () => {
    window.open(`${API_BASE_URL.replace('/api', '')}/api/admin/export/listings?format=csv`, '_blank');
  };

  const exportListingsJSON = () => {
    window.open(`${API_BASE_URL.replace('/api', '')}/api/admin/export/listings?format=json`, '_blank');
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Container maxWidth="xl">
        <Alert severity="error" action={<Button color="inherit" size="small" onClick={loadStats}>Retry</Button>}>
          {error}
        </Alert>
      </Container>
    );
  }

  const varietyChartData = stats?.variety_distribution
    ? Object.entries(stats.variety_distribution)
        .filter(([, c]) => c > 0)
        .map(([name, count]) => ({ name: name.replace(' rice', ''), count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 13)
    : [];
  const dateChartData = stats?.predictions_by_date?.slice(0, 14).reverse() || [];

  return (
    <Container maxWidth="xl">
      <Typography variant="h5" sx={{ mb: 2, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
        <AdminIcon /> Admin / Research dashboard
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Overview of predictions, marketplace listings, and purchase interests. Export data for research or reporting.
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" variant="body2">Total predictions</Typography>
                <Typography variant="h4">{stats?.total_predictions ?? 0}</Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" variant="body2">Total listings</Typography>
                <Typography variant="h4">{stats?.total_listings ?? 0}</Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" variant="body2">Purchase interests</Typography>
                <Typography variant="h4">{stats?.total_purchases ?? 0}</Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" variant="body2">Varieties predicted</Typography>
                <Typography variant="h4">
                  {stats?.variety_distribution ? Object.keys(stats.variety_distribution).filter(k => stats.variety_distribution[k] > 0).length : 0}
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      <Grid container spacing={3} sx={{ mt: 1 }}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <ChartIcon /> Predictions by variety
            </Typography>
            {varietyChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={varietyChartData} margin={{ top: 8, right: 8, left: 8, bottom: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="name" angle={-35} textAnchor="end" height={80} tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#90caf9" name="Predictions" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Typography color="text.secondary">No prediction data yet.</Typography>
            )}
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <TrendingIcon /> Predictions by date (last 14)
            </Typography>
            {dateChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={dateChartData} margin={{ top: 8, right: 8, left: 8, bottom: 24 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#f48fb1" name="Count" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Typography color="text.secondary">No date data yet.</Typography>
            )}
          </Paper>
        </Grid>
      </Grid>

      <Paper sx={{ p: 2, mt: 3 }}>
        <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ListIcon /> Listings by rice variety
        </Typography>
        {stats?.listings_by_variety?.length > 0 ? (
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell><strong>Rice variety</strong></TableCell>
                  <TableCell align="right"><strong>Listings</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {stats.listings_by_variety.map((row, i) => (
                  <TableRow key={i}>
                    <TableCell>{row.rice_name}</TableCell>
                    <TableCell align="right">{row.count}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Typography color="text.secondary">No listings yet.</Typography>
        )}
      </Paper>

      <Box sx={{ mt: 3 }}>
        <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <DownloadIcon /> Export data
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
          <Button variant="outlined" startIcon={<DownloadIcon />} onClick={exportPurchases}>
            Export purchases (CSV)
          </Button>
          <Button variant="outlined" startIcon={<DownloadIcon />} onClick={exportListingsCSV}>
            Export listings (CSV)
          </Button>
          <Button variant="outlined" startIcon={<DownloadIcon />} onClick={exportListingsJSON}>
            Export listings (JSON)
          </Button>
        </Box>
      </Box>

      <Button variant="text" size="small" onClick={loadStats} sx={{ mt: 2 }}>
        Refresh stats
      </Button>
    </Container>
  );
}

export default AdminPage;
